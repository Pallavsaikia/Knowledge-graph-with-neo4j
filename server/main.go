package main

import (
	"encoding/json"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

type Client struct {
	conn *websocket.Conn
	room string
}

type ServerInfo struct {
	Address string `json:"address"`
	Port    int    `json:"port"`
}

var (
	rooms     = make(map[string][]*Client)
	roomsMu   sync.Mutex
	servers   []ServerInfo
	serversMu sync.Mutex
	rnd       = rand.New(rand.NewSource(time.Now().UnixNano()))
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	roomId := r.URL.Query().Get("room")
	if roomId == "" {
		http.Error(w, "room query param required", http.StatusBadRequest)
		return
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade error:", err)
		return
	}

	client := &Client{conn: conn, room: roomId}

	roomsMu.Lock()
	rooms[roomId] = append(rooms[roomId], client)
	roomsMu.Unlock()

	log.Printf("Client joined room: %s", roomId)

	for {
		messageType, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("Read error (room %s): %v", roomId, err)
			break
		}

		broadcastToRoom(roomId, client, messageType, msg)
	}

	// Remove client on disconnect
	roomsMu.Lock()
	clients := rooms[roomId]
	for i, c := range clients {
		if c == client {
			rooms[roomId] = append(clients[:i], clients[i+1:]...)
			break
		}
	}
	roomsMu.Unlock()

	log.Printf("Client left room: %s", roomId)
	conn.Close()
}

func broadcastToRoom(roomId string, sender *Client, messageType int, msg []byte) {
	roomsMu.Lock()
	defer roomsMu.Unlock()
	for _, client := range rooms[roomId] {
		if client != sender {
			err := client.conn.WriteMessage(messageType, msg)
			if err != nil {
				log.Println("Write error:", err)
			}
		}
	}
}


func handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	var newServer ServerInfo
	if err := json.NewDecoder(r.Body).Decode(&newServer); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	serversMu.Lock()
	defer serversMu.Unlock()

	for _, s := range servers {
		if s.Address == newServer.Address && s.Port == newServer.Port {
			http.Error(w, "Already registered", http.StatusConflict)
			return
		}
	}

	servers = append(servers, newServer)
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(newServer)
}

func handleAllocate(w http.ResponseWriter, r *http.Request) {
	serversMu.Lock()
	defer serversMu.Unlock()

	if len(servers) == 0 {
		http.Error(w, "No servers available", http.StatusServiceUnavailable)
		return
	}

	selected := servers[rnd.Intn(len(servers))]
	json.NewEncoder(w).Encode(selected)
}

func handleList(w http.ResponseWriter, r *http.Request) {
	serversMu.Lock()
	defer serversMu.Unlock()
	json.NewEncoder(w).Encode(servers)
}

func main() {
	http.HandleFunc("/ws", handleWebSocket)
	http.HandleFunc("/register", handleRegister)
	http.HandleFunc("/allocate", handleAllocate)
	http.HandleFunc("/list", handleList)

	log.Println("Server + Registry running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
