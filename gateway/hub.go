// hub.go
// WebSocket 连接池与消息推送
//
// 架构：
//   Hub 维护所有活跃连接（map[*Client]bool）
//   Client 对应一个 WebSocket 连接，有独立的 send channel
//   Broadcast(msg) → 遍历所有 client，写入 send channel
//   客户端断开时自动从 Hub 移除
//
// 消息格式（JSON）：
//   {"type": "comment", "paper": "xxx.pdf", "content": "..."}
//   {"type": "recommend", "paper": "xxx.pdf"}
//   {"type": "ping"}

package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// ── WebSocket 升级器 ─────────────────────────────────────
var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true }, // 开发环境允许跨域
}

// ── 消息结构 ─────────────────────────────────────────────
type WSMessage struct {
	Type    string `json:"type"`
	Paper   string `json:"paper,omitempty"`
	Content string `json:"content,omitempty"`
	Data    any    `json:"data,omitempty"`
}

// ── Client：单条 WebSocket 连接 ─────────────────────────
type Client struct {
	hub  *Hub
	conn *websocket.Conn
	send chan []byte
}

// writePump：从 send channel 读消息，写到 WebSocket
func (c *Client) writePump() {
	ticker := time.NewTicker(30 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()
	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-ticker.C:
			// 心跳 ping
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// readPump：读客户端消息（保持连接存活），断开时注销
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()
	c.conn.SetReadLimit(512)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})
	for {
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			break
		}
	}
}

// ── Hub：连接池管理 ──────────────────────────────────────
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

func (h *Hub) run() {
	for {
		select {
		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = true
			h.mu.Unlock()
			log.Printf("[WS Hub] 新连接，当前 %d 个客户端", len(h.clients))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
			}
			h.mu.Unlock()
			log.Printf("[WS Hub] 连接断开，剩余 %d 个客户端", len(h.clients))

		case msg := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- msg:
				default:
					// send buffer 满 → 关闭该连接
					close(c.send)
					delete(h.clients, c)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// Broadcast 供外部调用（HTTP handler 触发推送）
func (h *Hub) Broadcast(msg WSMessage) {
	b, err := json.Marshal(msg)
	if err != nil {
		return
	}
	select {
	case h.broadcast <- b:
	default:
		log.Println("[WS Hub] broadcast channel 满，丢弃消息")
	}
}

// ConnCount 返回当前连接数
func (h *Hub) ConnCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.clients)
}

// ── HTTP Handler：升级为 WebSocket ─────────────────────
func wsServeHandler(hub *Hub) gin.HandlerFunc {
	return func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			log.Printf("[WS] upgrade error: %v", err)
			return
		}
		client := &Client{hub: hub, conn: conn, send: make(chan []byte, 64)}
		hub.register <- client

		go client.writePump()
		go client.readPump()
	}
}

// ── HTTP Handler：触发广播（供内部服务调用）────────────
// POST /internal/broadcast  body: {"type":"comment","paper":"x.pdf","content":"..."}
func broadcastHandler(hub *Hub) gin.HandlerFunc {
	return func(c *gin.Context) {
		var msg WSMessage
		if err := c.ShouldBindJSON(&msg); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		hub.Broadcast(msg)
		c.JSON(http.StatusOK, gin.H{
			"ok":      true,
			"clients": hub.ConnCount(),
		})
	}
}
