// main_test.go
// Go Gin 网关单元测试
//
// 运行：
//   cd gateway && go test ./... -v

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ── 工具：构建测试路由 ───────────────────────────────────
func newTestRouter() *gin.Engine {
	r := gin.New()
	rl := newRateLimiter()
	r.Use(rateLimitMiddleware(rl))
	r.Use(metricsMiddleware())
	r.Use(jwtMiddleware())
	r.GET("/health",  healthHandler)
	r.GET("/metrics", metricsHandler)
	return r
}

// ── /health ──────────────────────────────────────────────

func TestHealthEndpoint(t *testing.T) {
	r := newTestRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("health: got %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), "ok") {
		t.Errorf("health body missing 'ok': %s", w.Body.String())
	}
}

// ── 限流 ─────────────────────────────────────────────────

func TestRateLimiterAllowsUnderLimit(t *testing.T) {
	rl := newRateLimiter()
	for i := 0; i < 20; i++ {
		if !rl.allow("1.2.3.4") {
			t.Errorf("request %d should be allowed", i)
		}
	}
}

func TestRateLimiterBlocksOverLimit(t *testing.T) {
	rl := newRateLimiter()
	// 耗尽令牌桶
	for i := 0; i < 20; i++ {
		rl.allow("5.6.7.8")
	}
	if rl.allow("5.6.7.8") {
		t.Error("21st request should be blocked")
	}
}

func TestRateLimiterRefillsOverTime(t *testing.T) {
	rl := newRateLimiter()
	for i := 0; i < 20; i++ {
		rl.allow("9.9.9.9")
	}
	// 等待 100ms，令牌桶应补充约 2 个令牌（rate=20/s）
	time.Sleep(110 * time.Millisecond)
	if !rl.allow("9.9.9.9") {
		t.Error("token should refill after wait")
	}
}

func TestRateLimitMiddlewareReturns429(t *testing.T) {
	r := gin.New()
	rl := newRateLimiter()
	// 先耗尽（IP 与 ClientIP() 解析结果一致：127.0.0.1）
	for i := 0; i < 20; i++ {
		rl.allow("127.0.0.1")
	}
	r.Use(rateLimitMiddleware(rl))
	r.GET("/test", func(c *gin.Context) { c.Status(200) })

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	req.RemoteAddr = "127.0.0.1:9999"
	r.ServeHTTP(w, req)
	if w.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", w.Code)
	}
}

// ── JWT 中间件 ────────────────────────────────────────────

func makeToken(sub string) string {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"sub": sub,
		"exp": time.Now().Add(time.Hour).Unix(),
	})
	s, _ := token.SignedString(JWTSecret)
	return s
}

func TestJWTMiddlewareValidToken(t *testing.T) {
	r := gin.New()
	r.Use(jwtMiddleware())
	r.GET("/me", func(c *gin.Context) {
		uid, exists := c.Get("user_id")
		if !exists {
			c.JSON(401, gin.H{"error": "no user_id"})
			return
		}
		c.JSON(200, gin.H{"user_id": uid})
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/me", nil)
	req.Header.Set("Authorization", "Bearer "+makeToken("42"))
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("valid token: got %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), "42") {
		t.Errorf("user_id not injected: %s", w.Body.String())
	}
}

func TestJWTMiddlewareNoToken(t *testing.T) {
	r := gin.New()
	r.Use(jwtMiddleware())
	r.GET("/open", func(c *gin.Context) { c.Status(200) })

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/open", nil)
	r.ServeHTTP(w, req)
	// 无 token 应放行（匿名用户）
	if w.Code != 200 {
		t.Errorf("no token should pass: got %d", w.Code)
	}
}

func TestJWTMiddlewareInvalidToken(t *testing.T) {
	r := gin.New()
	r.Use(jwtMiddleware())
	r.GET("/open", func(c *gin.Context) { c.Status(200) })

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/open", nil)
	req.Header.Set("Authorization", "Bearer invalid.token.here")
	r.ServeHTTP(w, req)
	// 无效 token 不拦截，由下游处理
	if w.Code != 200 {
		t.Errorf("invalid token should pass through: got %d", w.Code)
	}
}

// ── /metrics ─────────────────────────────────────────────

func TestMetricsEndpointReturnsText(t *testing.T) {
	r := newTestRouter()
	// 先打一个请求触发指标记录
	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w1, req1)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/metrics", nil)
	r.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("metrics: got %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), "gateway_requests_total") {
		t.Errorf("metrics missing counter: %s", w.Body.String())
	}
}

// ── min64 工具函数 ────────────────────────────────────────

func TestMin64(t *testing.T) {
	if min64(3, 5) != 3 {
		t.Error("min64(3,5) should be 3")
	}
	if min64(5, 3) != 3 {
		t.Error("min64(5,3) should be 3")
	}
	if min64(4, 4) != 4 {
		t.Error("min64(4,4) should be 4")
	}
}
