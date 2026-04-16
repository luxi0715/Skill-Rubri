// prom.go
// Go 网关 Prometheus 指标定义
//
// 指标：
//   gateway_requests_total       Counter   请求总数，按 path/method/status
//   gateway_request_duration_seconds Histogram 请求耗时，按 path
//   gateway_active_connections   Gauge     当前活跃 WebSocket 连接数
//   gateway_rate_limit_blocked   Counter   被限流拒绝的请求数
//   gateway_proxy_errors_total   Counter   反向代理失败次数

package main

import (
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "HTTP 请求总数",
		},
		[]string{"path", "method", "status"},
	)

	httpDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "HTTP 请求耗时（秒）",
			Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0},
		},
		[]string{"path"},
	)

	wsActiveConns = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "gateway_active_ws_connections",
			Help: "当前 WebSocket 活跃连接数",
		},
	)

	rateLimitBlocked = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "gateway_rate_limit_blocked_total",
			Help: "被限流拒绝的请求数",
		},
	)

	proxyErrors = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "gateway_proxy_errors_total",
			Help: "反向代理失败次数",
		},
	)
)

var promOnce sync.Once

func initPrometheus() {
	promOnce.Do(func() {
		prometheus.MustRegister(
			httpRequestsTotal,
			httpDuration,
			wsActiveConns,
			rateLimitBlocked,
			proxyErrors,
		)
	})
}

// promMiddleware：记录每个请求的耗时和状态码
func promMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		duration := time.Since(start).Seconds()
		status   := strconv.Itoa(c.Writer.Status())
		path     := c.FullPath()
		if path == "" {
			path = "proxied"   // NoRoute 的请求
		}
		httpRequestsTotal.WithLabelValues(path, c.Request.Method, status).Inc()
		httpDuration.WithLabelValues(path).Observe(duration)
	}
}

// promRateLimitMiddleware：限流时额外记录指标
func promRateLimitMiddleware(rl *rateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !rl.allow(c.ClientIP()) {
			rateLimitBlocked.Inc()
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": "rate limit exceeded",
			})
			return
		}
		c.Next()
	}
}

// /metrics 路由：标准 Prometheus 格式
func promHandler() gin.HandlerFunc {
	h := promhttp.Handler()
	return func(c *gin.Context) {
		h.ServeHTTP(c.Writer, c.Request)
	}
}
