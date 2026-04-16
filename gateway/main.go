// main.go
// Go Gin API 网关
//
// 职责：
//   - 统一 HTTP 入口，监听 :8080
//   - JWT 鉴权中间件（验证 token，注入 user_id）
//   - 限流中间件（每 IP 每秒最多 20 次请求）
//   - 反向代理：非鉴权路由转发给 FastAPI :8000
//   - WebSocket Hub：实时推送评论/推荐更新
//   - /metrics 暴露 Prometheus 指标
//
// 运行：
//   cd gateway && go run main.go

package main

import (
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

// ── 配置 ────────────────────────────────────────────────
var (
	FastAPIAddr = getEnv("FASTAPI_ADDR", "http://localhost:8000")
	JWTSecret   = []byte(getEnv("JWT_SECRET", "change-me-in-production"))
	GatewayPort = getEnv("GATEWAY_PORT", "8080")
)

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ── 限流（令牌桶，按 IP）────────────────────────────────
type rateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*tokenBucket
}

type tokenBucket struct {
	tokens   float64
	lastTime time.Time
	rate     float64 // tokens per second
	capacity float64
}

func newRateLimiter() *rateLimiter {
	return &rateLimiter{buckets: make(map[string]*tokenBucket)}
}

func (rl *rateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	b, ok := rl.buckets[ip]
	if !ok {
		b = &tokenBucket{tokens: 20, lastTime: time.Now(), rate: 20, capacity: 20}
		rl.buckets[ip] = b
	}

	now := time.Now()
	elapsed := now.Sub(b.lastTime).Seconds()
	b.tokens = min64(b.capacity, b.tokens+elapsed*b.rate)
	b.lastTime = now

	if b.tokens >= 1 {
		b.tokens--
		return true
	}
	return false
}

func min64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// ── Prometheus 指标（轻量手写，不引入 prometheus/client_go）──
var (
	metricsMu       sync.Mutex
	requestTotal    = map[string]int64{}
	requestDuration = map[string]float64{}
)

func recordMetric(path string, duration float64) {
	metricsMu.Lock()
	defer metricsMu.Unlock()
	requestTotal[path]++
	requestDuration[path] += duration
}

// ── 中间件：限流 ─────────────────────────────────────────
func rateLimitMiddleware(rl *rateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := c.ClientIP()
		if !rl.allow(ip) {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": "rate limit exceeded, max 20 req/s per IP",
			})
			return
		}
		c.Next()
	}
}

// ── 中间件：请求计时 + 指标收集 ─────────────────────────
func metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		duration := time.Since(start).Seconds()
		recordMetric(c.FullPath(), duration)
	}
}

// ── 中间件：JWT 鉴权（可选，Bearer token）───────────────
func jwtMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		auth := c.GetHeader("Authorization")
		if auth == "" {
			// 无 token → 匿名用户，不拦截，只是不注入 user_id
			c.Next()
			return
		}

		tokenStr := auth
		if len(auth) > 7 && auth[:7] == "Bearer " {
			tokenStr = auth[7:]
		}

		token, err := jwt.Parse(tokenStr, func(t *jwt.Token) (interface{}, error) {
			if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method")
			}
			return JWTSecret, nil
		})

		if err != nil || !token.Valid {
			// token 无效 → 继续但不注入，由下游 FastAPI 二次验证
			c.Next()
			return
		}

		if claims, ok := token.Claims.(jwt.MapClaims); ok {
			c.Set("user_id", claims["sub"])
		}
		c.Next()
	}
}

// ── 反向代理：转发给 FastAPI ─────────────────────────────
func reverseProxy(target string) gin.HandlerFunc {
	targetURL, err := url.Parse(target)
	if err != nil {
		log.Fatalf("invalid FASTAPI_ADDR: %v", err)
	}
	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	return func(c *gin.Context) {
		proxy.ServeHTTP(c.Writer, c.Request)
	}
}

// ── WebSocket Hub（见 hub.go）────────────────────────────
// wsHandler 和 broadcastHandler 在 hub.go 里定义
var globalHub = newHub()

// ── /metrics 接口 ────────────────────────────────────────
func metricsHandler(c *gin.Context) {
	metricsMu.Lock()
	defer metricsMu.Unlock()

	out := "# HELP gateway_requests_total Total requests per path\n"
	out += "# TYPE gateway_requests_total counter\n"
	for path, count := range requestTotal {
		out += fmt.Sprintf("gateway_requests_total{path=%q} %d\n", path, count)
	}
	out += "# HELP gateway_request_duration_seconds_total Total duration per path\n"
	out += "# TYPE gateway_request_duration_seconds_total counter\n"
	for path, dur := range requestDuration {
		out += fmt.Sprintf("gateway_request_duration_seconds_total{path=%q} %.6f\n", path, dur)
	}
	c.String(http.StatusOK, out)
}

// ── /health 健康检查 ─────────────────────────────────────
func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok", "gateway": "papergateway/v1"})
}

// ── 主函数 ───────────────────────────────────────────────
func main() {
	initPrometheus()

	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	rl := newRateLimiter()
	r.Use(promRateLimitMiddleware(rl))   // 限流 + Prometheus 指标
	r.Use(promMiddleware())              // 请求耗时指标
	r.Use(jwtMiddleware())

	// 启动 Hub goroutine
	go globalHub.run()

	// 网关自身接口
	r.GET("/health",              healthHandler)
	r.GET("/metrics",             promHandler())   // 标准 Prometheus 格式
	r.GET("/ws",                  wsServeHandler(globalHub))
	r.POST("/internal/broadcast", broadcastHandler(globalHub))

	// 其余所有请求转发给 FastAPI
	proxy := reverseProxy(FastAPIAddr)
	r.NoRoute(proxy)

	addr := ":" + GatewayPort
	log.Printf("[Gateway] 启动在 %s，转发至 %s", addr, FastAPIAddr)
	if err := r.Run(addr); err != nil {
		log.Fatalf("gateway error: %v", err)
	}
}
