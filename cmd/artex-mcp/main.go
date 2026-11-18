// Command artex-mcp is a stdio MCP server that exposes a running ARTEX
// instance's REST API as tools, so MCP clients (Claude Code, Claude Desktop,
// etc.) can create scoped tasks, steer the agent, decide approvals and pull
// findings/reports.
//
// It is a pure HTTP client: it never imports the server package and has no
// access ARTEX itself does not grant over /api.
//
// Configuration (environment):
//
//	ARTEX_URL       base URL (default http://127.0.0.1:8787)
//	ARTEX_PASSWORD  admin password; used to log in and re-login on 401
//	ARTEX_PASSWORD_FILE  alternatively, a file containing the password
//	ARTEX_TOKEN     alternatively, an existing JWT (7-day lifetime)
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const maxResultBytes = 200 << 10 // keep tool results within a sane context budget

var version = "dev"

// client is a minimal authenticated ARTEX API client.
type client struct {
	base     string
	password string
	http     *http.Client

	mu    sync.Mutex
	token string
}

func newClient() *client {
	base := strings.TrimRight(os.Getenv("ARTEX_URL"), "/")
	if base == "" {
		base = "http://127.0.0.1:8787"
	}
	pw := os.Getenv("ARTEX_PASSWORD")
	if f := os.Getenv("ARTEX_PASSWORD_FILE"); pw == "" && f != "" {
		b, err := os.ReadFile(f)
		if err != nil {
			log.Fatalf("ARTEX_PASSWORD_FILE: %v", err)
		}
		pw = strings.TrimSpace(string(b))
	}
	return &client{
		base:     base,
		password: pw,
		token:    os.Getenv("ARTEX_TOKEN"),
		http:     &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *client) login(ctx context.Context) error {
	if c.password == "" {
		return errors.New("not authenticated: set ARTEX_PASSWORD / ARTEX_PASSWORD_FILE (or a fresh ARTEX_TOKEN) in the MCP server env")
	}
	body, _ := json.Marshal(map[string]string{"username": "ARTEX", "password": c.password})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/api/auth/login", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("login: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("login failed: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var out struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || out.Token == "" {
		return fmt.Errorf("login: unexpected response: %s", strings.TrimSpace(string(raw)))
	}
	c.token = out.Token
	return nil
}

// do performs an API call, logging in lazily and retrying once on 401.
func (c *client) do(ctx context.Context, method, path string, q url.Values, body any) (int, []byte, error) {
	var payload []byte
	if body != nil {
		var err error
		if payload, err = json.Marshal(body); err != nil {
			return 0, nil, err
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for attempt := 0; attempt < 2; attempt++ {
		if c.token == "" {
			if err := c.login(ctx); err != nil {
				return 0, nil, err
			}
		}
		u := c.base + path
		if len(q) > 0 {
			u += "?" + q.Encode()
		}
		var rd io.Reader
		if payload != nil {
			rd = bytes.NewReader(payload)
		}
		req, err := http.NewRequestWithContext(ctx, method, u, rd)
		if err != nil {
			return 0, nil, err
		}
		if payload != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		req.Header.Set("Authorization", "Bearer "+c.token)
		resp, err := c.http.Do(req)
		if err != nil {
			return 0, nil, err
		}
		raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		resp.Body.Close()
		if err != nil {
			return resp.StatusCode, nil, err
		}
		if resp.StatusCode == http.StatusUnauthorized && attempt == 0 && c.password != "" {
			c.token = ""
			continue
		}
		return resp.StatusCode, raw, nil
	}
	return 0, nil, errors.New("unauthorized after re-login")
}

// call wraps do() into an MCP tool result. Non-2xx responses become tool
// errors (IsError) so the model sees the API's message instead of a crash.
func (c *client) call(ctx context.Context, method, path string, q url.Values, body any) (*mcp.CallToolResult, any, error) {
	status, raw, err := c.do(ctx, method, path, q, body)
	if err != nil {
		return errResult(err.Error()), nil, nil
	}
	text := prettify(raw)
	if status < 200 || status > 299 {
		return errResult(fmt.Sprintf("%s %s -> HTTP %d\n%s", method, path, status, text)), nil, nil
	}
	if text == "" {
		text = fmt.Sprintf("HTTP %d (empty body)", status)
	}
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: text}}}, nil, nil
}

func errResult(msg string) *mcp.CallToolResult {
	return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: msg}}}
}

func prettify(raw []byte) string {
	var buf bytes.Buffer
	if json.Valid(raw) && json.Indent(&buf, raw, "", " ") == nil {
		raw = buf.Bytes()
	}
	s := string(raw)
	if len(s) > maxResultBytes {
		s = s[:maxResultBytes] + fmt.Sprintf("\n…[truncated: %d of %d bytes shown; narrow the query]", maxResultBytes, len(raw))
	}
	return s
}

func setIf(q url.Values, k, v string) {
	if v != "" {
		q.Set(k, v)
	}
}

func setIntIf(q url.Values, k string, v int) {
	if v > 0 {
		q.Set(k, strconv.Itoa(v))
	}
}

func esc(s string) string { return url.PathEscape(s) }

// ---- tool inputs ----

type noArgs struct{}

type scopeItem struct {
	Kind  string `json:"kind" jsonschema:"one of: root_domain, subdomain, ip, cidr, icp, keyword"`
	Value string `json:"value" jsonschema:"the asset, e.g. 192.168.1.0/24 or example.com"`
}

type createCompanyIn struct {
	Name  string      `json:"name" jsonschema:"company / engagement name (e.g. a code word)"`
	Scope []scopeItem `json:"scope,omitempty" jsonschema:"initial in-scope assets"`
}

type companyScopeIn struct {
	CompanyID int64       `json:"company_id"`
	Scope     []scopeItem `json:"scope"`
	Reason    string      `json:"reason" jsonschema:"why this is authorized / in scope (recorded for audit)"`
	Reset     bool        `json:"reset,omitempty" jsonschema:"replace the existing scope instead of appending"`
}

type taskIDIn struct {
	TaskID string `json:"task_id"`
}

type createTaskIn struct {
	Name           string  `json:"name"`
	Goal           string  `json:"goal" jsonschema:"what the agent should achieve, e.g. 'discover and map all live hosts and services'"`
	Description    string  `json:"description,omitempty"`
	CompanyIDs     []int64 `json:"company_ids,omitempty" jsonschema:"companies whose scope this task inherits; set these so the task starts already scoped"`
	TimeoutSeconds int     `json:"timeout_seconds,omitempty"`
	LLMProfileID   string  `json:"llm_profile_id,omitempty" jsonschema:"optional LLM profile; defaults to the instance default"`
}

type taskScopeIn struct {
	TaskID string `json:"task_id"`
	Kind   string `json:"kind" jsonschema:"one of: company, root_domain, subdomain, ip, cidr, icp, keyword"`
	Value  string `json:"value"`
	Reason string `json:"reason" jsonschema:"why this is authorized / in scope"`
}

type controlIn struct {
	TaskID string `json:"task_id"`
	Action string `json:"action" jsonschema:"pause or resume"`
}

type chatIn struct {
	TaskID  string `json:"task_id"`
	Message string `json:"message" jsonschema:"instruction or question for the task's main agent"`
}

type activityIn struct {
	TaskID string `json:"task_id"`
	Since  int    `json:"since,omitempty" jsonschema:"only events after this sequence number (use the last seq you saw)"`
	Limit  int    `json:"limit,omitempty" jsonschema:"max events (default 50)"`
}

type decideIn struct {
	ID       string `json:"id" jsonschema:"pending approval id"`
	Decision string `json:"decision" jsonschema:"allowed or denied"`
}

type findingsIn struct {
	TaskID   string `json:"task_id,omitempty"`
	Severity string `json:"severity,omitempty" jsonschema:"critical, high, medium, low or info"`
	Status   string `json:"status,omitempty"`
	Q        string `json:"q,omitempty" jsonschema:"free-text search"`
	Page     int    `json:"page,omitempty"`
	Limit    int    `json:"limit,omitempty" jsonschema:"page size (default 50)"`
}

type idIn struct {
	ID string `json:"id"`
}

type assetsIn struct {
	Query string `json:"query,omitempty" jsonschema:"raw query string passed to GET /api/assets, e.g. 'type=host&q=192.168'"`
}

type apiIn struct {
	Method string          `json:"method" jsonschema:"GET, POST, PATCH, PUT or DELETE"`
	Path   string          `json:"path" jsonschema:"API path starting with /api/"`
	Query  string          `json:"query,omitempty" jsonschema:"raw query string without the leading ?"`
	Body   json.RawMessage `json:"body,omitempty" jsonschema:"JSON request body"`
}

func main() {
	log.SetOutput(os.Stderr) // stdout is the MCP channel
	c := newClient()

	s := mcp.NewServer(&mcp.Implementation{Name: "artex", Version: version}, &mcp.ServerOptions{
		Instructions: "Drives a self-hosted ARTEX pentest instance. Only target assets the operator owns or is authorized to test. " +
			"Typical flow: artex_create_company (with scope + reason) -> artex_create_task (company_ids; it launches immediately) -> " +
			"artex_activity / artex_pending_approvals / artex_decide_approval -> artex_list_findings -> artex_report.",
	})

	ro := &mcp.ToolAnnotations{ReadOnlyHint: true}

	mcp.AddTool(s, &mcp.Tool{Name: "artex_status", Description: "Check ARTEX health and that authentication works.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/auth/status", nil, nil)
		})

	// companies / scope
	mcp.AddTool(s, &mcp.Tool{Name: "artex_list_companies", Description: "List companies (engagement containers) and their scope.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/companies", nil, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_create_company", Description: "Create a company (engagement container), optionally with initial in-scope assets."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in createCompanyIn) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "POST", "/api/companies", nil, in)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_add_company_scope", Description: "Add (or with reset, replace) in-scope assets on a company, recording the authorization reason."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in companyScopeIn) (*mcp.CallToolResult, any, error) {
			body := map[string]any{"scope": in.Scope, "reason": in.Reason, "reset": in.Reset}
			return c.call(ctx, "POST", fmt.Sprintf("/api/companies/%d/scope", in.CompanyID), nil, body)
		})

	// tasks
	mcp.AddTool(s, &mcp.Tool{Name: "artex_list_tasks", Description: "List tasks with status, goal progress and finding counts.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/tasks", nil, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_get_task", Description: "Get one task's details.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in taskIDIn) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/tasks/"+esc(in.TaskID), nil, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_create_task", Description: "Create a task. It LAUNCHES IMMEDIATELY, so pass company_ids for a company whose scope is already set."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in createTaskIn) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "POST", "/api/tasks", nil, in)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_list_task_scope", Description: "List a task's scope entries.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in taskIDIn) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/tasks/"+esc(in.TaskID)+"/scope", nil, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_add_task_scope", Description: "Add one scope entry directly to a task, with the authorization reason."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in taskScopeIn) (*mcp.CallToolResult, any, error) {
			body := map[string]string{"kind": in.Kind, "value": in.Value, "reason": in.Reason}
			return c.call(ctx, "POST", "/api/tasks/"+esc(in.TaskID)+"/scope", nil, body)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_control_task", Description: "Pause or resume a task."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in controlIn) (*mcp.CallToolResult, any, error) {
			if in.Action != "pause" && in.Action != "resume" {
				return errResult("action must be pause or resume"), nil, nil
			}
			return c.call(ctx, "POST", "/api/tasks/"+esc(in.TaskID)+"/control", nil, map[string]string{"action": in.Action})
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_chat", Description: "Send a message to a task's main agent. LLM replies arrive asynchronously in artex_activity."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in chatIn) (*mcp.CallToolResult, any, error) {
			q := url.Values{"task": {in.TaskID}}
			return c.call(ctx, "POST", "/api/chat", q, map[string]string{"message": in.Message})
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_activity", Description: "Read a task's recent agent activity (planner/worker events, chat replies). Poll with since=<last seq>.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in activityIn) (*mcp.CallToolResult, any, error) {
			if in.Limit <= 0 {
				in.Limit = 50
			}
			q := url.Values{}
			setIf(q, "task", in.TaskID)
			setIntIf(q, "since", in.Since)
			setIntIf(q, "limit", in.Limit)
			return c.call(ctx, "GET", "/api/exploration/activity", q, nil)
		})

	// approvals
	mcp.AddTool(s, &mcp.Tool{Name: "artex_pending_approvals", Description: "List tool executions waiting for human approval.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, _ noArgs) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/intercept/pending", nil, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_decide_approval", Description: "Allow or deny one pending tool execution. Confirm with the operator before allowing anything outside the stated scope."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in decideIn) (*mcp.CallToolResult, any, error) {
			if in.Decision != "allowed" && in.Decision != "denied" {
				return errResult("decision must be allowed or denied"), nil, nil
			}
			return c.call(ctx, "POST", "/api/intercept/pending/"+esc(in.ID)+"/decide", nil, map[string]string{"decision": in.Decision})
		})

	// results
	mcp.AddTool(s, &mcp.Tool{Name: "artex_list_assets", Description: "List discovered assets (hosts, services, URLs...).", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in assetsIn) (*mcp.CallToolResult, any, error) {
			q, err := url.ParseQuery(in.Query)
			if err != nil {
				return errResult("bad query: " + err.Error()), nil, nil
			}
			return c.call(ctx, "GET", "/api/assets", q, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_list_findings", Description: "List findings, optionally for one task and filtered by severity/status/text.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in findingsIn) (*mcp.CallToolResult, any, error) {
			if in.Page <= 0 {
				in.Page = 1
			}
			if in.Limit <= 0 {
				in.Limit = 50
			}
			q := url.Values{}
			setIf(q, "task_id", in.TaskID)
			setIf(q, "severity", in.Severity)
			setIf(q, "status", in.Status)
			setIf(q, "q", in.Q)
			setIntIf(q, "page", in.Page)
			setIntIf(q, "limit", in.Limit)
			return c.call(ctx, "GET", "/api/exploration/findings", q, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_get_finding", Description: "Get one finding with its evidence and report text.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in idIn) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/exploration/findings/"+esc(in.ID), nil, nil)
		})
	mcp.AddTool(s, &mcp.Tool{Name: "artex_report", Description: "Render a task's markdown report.", Annotations: ro},
		func(ctx context.Context, _ *mcp.CallToolRequest, in taskIDIn) (*mcp.CallToolResult, any, error) {
			return c.call(ctx, "GET", "/api/report", url.Values{"task": {in.TaskID}}, nil)
		})

	// escape hatch
	mcp.AddTool(s, &mcp.Tool{Name: "artex_api", Description: "Call any ARTEX /api endpoint directly (for anything the other tools don't cover)."},
		func(ctx context.Context, _ *mcp.CallToolRequest, in apiIn) (*mcp.CallToolResult, any, error) {
			m := strings.ToUpper(in.Method)
			switch m {
			case "GET", "POST", "PATCH", "PUT", "DELETE":
			default:
				return errResult("unsupported method " + in.Method), nil, nil
			}
			if !strings.HasPrefix(in.Path, "/api/") {
				return errResult("path must start with /api/"), nil, nil
			}
			q, err := url.ParseQuery(in.Query)
			if err != nil {
				return errResult("bad query: " + err.Error()), nil, nil
			}
			var body any
			if len(in.Body) > 0 && string(in.Body) != "null" {
				body = in.Body
			}
			return c.call(ctx, m, in.Path, q, body)
		})

	if err := s.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}
