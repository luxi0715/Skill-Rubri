// prom_test.go
// Prometheus 指标测试
//
// 运行：cd gateway && go test ./... -v

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
)

func init() {
	gin.SetMode(gin.TestMode)
	initPrometheus()   // 注册指标，sync.Once 保证只跑一次
}

// 每个测试用独立的 Registry 避免重复注册
func newTestRegistry() *prometheus.Registry {
	return prometheus.NewRegistry()
}

func TestPromMetricsEndpointFormat(t *testing.T) {
	// 用全局指标（已在 prom.go 注册）
	r := gin.New()
	r.GET("/metrics", promHandler())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/metrics", nil)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("metrics: got %d, want 200", w.Code)
	}
	body := w.Body.String()
	// Prometheus 格式必须包含 # HELP 和 # TYPE
	if !strings.Contains(body, "# HELP") {
		t.Error("metrics missing # HELP")
	}
	if !strings.Contains(body, "# TYPE") {
		t.Error("metrics missing # TYPE")
	}
}

func TestPromMetricsContainsGatewayMetrics(t *testing.T) {
	r := gin.New()
	r.Use(promMiddleware())
	r.GET("/health", healthHandler)
	r.GET("/metrics", promHandler())

	// 先打一个请求触发指标
	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("GET", "/health", nil)
	r.ServeHTTP(w1, req1)

	// 查指标
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/metrics", nil)
	r.ServeHTTP(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "gateway_requests_total") {
		t.Error("missing gateway_requests_total")
	}
	if !strings.Contains(body, "gateway_request_duration_seconds") {
		t.Error("missing gateway_request_duration_seconds")
	}
}

func TestRateLimitBlockedMetric(t *testing.T) {
	r := gin.New()
	rl := newRateLimiter()
	// 耗尽令牌
	for i := 0; i < 20; i++ {
		rl.allow("10.0.0.1")
	}
	r.Use(promRateLimitMiddleware(rl))
	r.GET("/test", func(c *gin.Context) { c.Status(200) })
	r.GET("/metrics", promHandler())

	// 触发限流
	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("GET", "/test", nil)
	req1.RemoteAddr = "10.0.0.1:1234"
	r.ServeHTTP(w1, req1)

	if w1.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", w1.Code)
	}

	// 验证指标里有 rate_limit
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/metrics", nil)
	r.ServeHTTP(w, req)
	if !strings.Contains(w.Body.String(), "gateway_rate_limit_blocked_total") {
		t.Error("missing rate_limit_blocked_total metric")
	}
}

func TestWSActiveConnectionsGauge(t *testing.T) {
	// wsActiveConns.Set(n) 后指标值应反映
	wsActiveConns.Set(5)
	r := gin.New()
	r.GET("/metrics", promHandler())

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/metrics", nil)
	r.ServeHTTP(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "gateway_active_ws_connections") {
		t.Error("missing ws_connections gauge")
	}
	wsActiveConns.Set(0) // 清理
}

func TestPromMiddlewareLabelsPath(t *testing.T) {
	r := gin.New()
	r.Use(promMiddleware())
	r.GET("/paper/:name", func(c *gin.Context) { c.Status(200) })
	r.GET("/metrics", promHandler())

	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("GET", "/paper/test", nil)
	r.ServeHTTP(w1, req1)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/metrics", nil)
	r.ServeHTTP(w, req)

	// path label 应为路由模板 /paper/:name，不是实际路径
	if !strings.Contains(w.Body.String(), "/paper/:name") {
		t.Error("path label should be route template, not actual path")
	}
}
