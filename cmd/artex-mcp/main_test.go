package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func text(r *mcp.CallToolResult) string { return r.Content[0].(*mcp.TextContent).Text }

func TestClientLoginAndRelogin(t *testing.T) {
	var logins atomic.Int32
	valid := "tok-1"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/auth/login":
			var in map[string]string
			_ = json.NewDecoder(r.Body).Decode(&in)
			if in["username"] != "ARTEX" || in["password"] != "pw" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			n := logins.Add(1)
			valid = "tok-" + string(rune('0'+n))
			_ = json.NewEncoder(w).Encode(map[string]string{"token": valid})
		case "/api/tasks":
			if r.Header.Get("Authorization") != "Bearer "+valid {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			_, _ = w.Write([]byte(`[{"id":"t1"}]`))
		case "/api/boom":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"bad scope"}`))
		}
	}))
	defer srv.Close()

	c := &client{base: srv.URL, password: "pw", token: "stale", http: srv.Client()}
	ctx := context.Background()

	res, _, _ := c.call(ctx, "GET", "/api/tasks", nil, nil)
	if res.IsError || !strings.Contains(text(res), `"t1"`) {
		t.Fatalf("expected tasks after re-login, got %q", text(res))
	}
	if logins.Load() != 1 {
		t.Fatalf("expected 1 login (stale token refresh), got %d", logins.Load())
	}

	res, _, _ = c.call(ctx, "GET", "/api/tasks", nil, nil)
	if res.IsError || logins.Load() != 1 {
		t.Fatalf("token should be reused; logins=%d", logins.Load())
	}

	res, _, _ = c.call(ctx, "POST", "/api/boom", nil, map[string]string{"x": "y"})
	if !res.IsError || !strings.Contains(text(res), "HTTP 400") || !strings.Contains(text(res), "bad scope") {
		t.Fatalf("expected tool error with API message, got %q", text(res))
	}
}

func TestClientNoCredentials(t *testing.T) {
	c := &client{base: "http://127.0.0.1:1", http: http.DefaultClient}
	res, _, _ := c.call(context.Background(), "GET", "/api/tasks", nil, nil)
	if !res.IsError || !strings.Contains(text(res), "ARTEX_PASSWORD") {
		t.Fatalf("expected credentials hint, got %q", text(res))
	}
}
