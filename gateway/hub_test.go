// hub_test.go
// WebSocket Hub 单元测试
//
// 运行：
//   cd gateway && go test ./... -v

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// ── Hub 基础功能 ─────────────────────────────────────────

func TestHubStartsWithZeroClients(t *testing.T) {
	h := newHub()
	go h.run()
	if h.ConnCount() != 0 {
		t.Errorf("new hub should have 0 clients, got %d", h.ConnCount())
	}
}

func TestHubBroadcastDoesNotPanicWhenEmpty(t *testing.T) {
	h := newHub()
	go h.run()
	// 无客户端时 Broadcast 不应 panic
	h.Broadcast(WSMessage{Type: "ping"})
}

func TestWSMessageJSON(t *testing.T) {
	msg := WSMessage{Type: "comment", Paper: "a.pdf", Content: "hello"}
	b, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, "comment") {
		t.Error("JSON missing type")
	}
	if !strings.Contains(s, "a.pdf") {
		t.Error("JSON missing paper")
	}
}

func TestWSMessageOmitsEmptyFields(t *testing.T) {
	msg := WSMessage{Type: "ping"}
	b, _ := json.Marshal(msg)
	s := string(b)
	if strings.Contains(s, "paper") {
		t.Error("empty paper field should be omitted")
	}
}

// ── 广播 Handler ─────────────────────────────────────────

func TestBroadcastHandlerReturnsOK(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := newHub()
	go h.run()

	r := gin.New()
	r.POST("/internal/broadcast", broadcastHandler(h))

	body := `{"type":"comment","paper":"test.pdf","content":"hello"}`
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/internal/broadcast",
		strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("broadcast: got %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), "true") {
		t.Errorf("broadcast: missing ok:true in %s", w.Body.String())
	}
}

func TestBroadcastHandlerBadJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := newHub()
	go h.run()

	r := gin.New()
	r.POST("/internal/broadcast", broadcastHandler(h))

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/internal/broadcast",
		strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("bad json: got %d, want 400", w.Code)
	}
}

// ── WebSocket 连接测试 ────────────────────────────────────

func TestWSConnectAndReceiveBroadcast(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := newHub()
	go h.run()

	r := gin.New()
	r.GET("/ws", wsServeHandler(h))
	r.POST("/internal/broadcast", broadcastHandler(h))

	srv := httptest.NewServer(r)
	defer srv.Close()

	// 建立 WebSocket 连接
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("ws dial error: %v", err)
	}
	defer conn.Close()

	time.Sleep(50 * time.Millisecond) // 等注册完成

	// 验证连接数
	if h.ConnCount() != 1 {
		t.Errorf("expected 1 client, got %d", h.ConnCount())
	}

	// 广播一条消息
	h.Broadcast(WSMessage{Type: "comment", Paper: "a.pdf", Content: "test msg"})

	// 读取消息
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, msg, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("read error: %v", err)
	}
	if !strings.Contains(string(msg), "test msg") {
		t.Errorf("expected 'test msg' in message, got %s", msg)
	}
}

func TestWSClientCountDecreaseOnDisconnect(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := newHub()
	go h.run()

	r := gin.New()
	r.GET("/ws", wsServeHandler(h))

	srv := httptest.NewServer(r)
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial error: %v", err)
	}

	time.Sleep(50 * time.Millisecond)
	if h.ConnCount() != 1 {
		t.Errorf("expected 1 client after connect, got %d", h.ConnCount())
	}

	conn.Close()
	time.Sleep(100 * time.Millisecond) // 等 unregister 处理
	if h.ConnCount() != 0 {
		t.Errorf("expected 0 clients after disconnect, got %d", h.ConnCount())
	}
}
