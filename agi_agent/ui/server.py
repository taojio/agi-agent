"""
ui/server.py - AGI Agent Web Server

提供 REST API 和 WebSocket 服务，支持实时交互和监控
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("agi_agent.ui")


class AGIAgentServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._server = None
        self._running = False
        self._chat_handler = None
        self._task_tracker = None
        self._security_dashboard = None
        self._websocket_manager = None
        self._agent_ref = None

        self._routes = {
            "/api/chat/send": self._handle_chat_send,
            "/api/chat/history": self._handle_chat_history,
            "/api/chat/sessions": self._handle_chat_sessions,
            "/api/chat/create": self._handle_chat_create,
            "/api/tasks/list": self._handle_tasks_list,
            "/api/tasks/create": self._handle_tasks_create,
            "/api/tasks/status": self._handle_tasks_status,
            "/api/tasks/update": self._handle_tasks_update,
            "/api/security/overview": self._handle_security_overview,
            "/api/security/alerts": self._handle_security_alerts,
            "/api/security/risk": self._handle_security_risk,
            "/api/system/status": self._handle_system_status,
        }

    def set_chat_handler(self, chat_handler):
        self._chat_handler = chat_handler

    def set_task_tracker(self, task_tracker):
        self._task_tracker = task_tracker

    def set_security_dashboard(self, security_dashboard):
        self._security_dashboard = security_dashboard

    def set_websocket_manager(self, websocket_manager):
        self._websocket_manager = websocket_manager

    def set_agent_ref(self, agent_ref):
        self._agent_ref = agent_ref
        if self._chat_handler:
            self._chat_handler.set_agent_ref(agent_ref)

    async def start(self):
        self._running = True
        logger.info(f"Starting AGI Agent Server on {self.host}:{self.port}")
        
        server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port
        )
        
        addr = server.sockets[0].getsockname()
        logger.info(f"Server listening on {addr}")
        
        async with server:
            await server.serve_forever()

    def run(self):
        asyncio.run(self.start())

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            data = await reader.read(8192)
            request = data.decode("utf-8", errors="replace")
            
            if not request:
                writer.close()
                await writer.wait_closed()
                return

            lines = request.split("\r\n")
            if not lines:
                writer.close()
                await writer.wait_closed()
                return

            first_line = lines[0].split(" ")
            if len(first_line) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method = first_line[0]
            path = first_line[1]

            parsed = urlparse(path)
            path = parsed.path
            query_params = parse_qs(parsed.query)

            content_type = "text/plain"
            body = b""
            
            for line in lines:
                if line.lower().startswith("content-type:"):
                    content_type = line.split(":")[1].strip()
            
            content_length = 0
            for line in lines:
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":")[1].strip())
                    break
            
            if content_length > 0:
                body = data[-content_length:]

            if path in self._routes:
                handler = self._routes[path]
                response = handler(method, query_params, body)
            elif path == "/ws":
                await self._handle_websocket(reader, writer)
                return
            elif path == "/" or path == "/index.html":
                response = self._handle_index()
            else:
                response = self._handle_not_found()

            status_code, status_msg, headers, response_body = response
            
            header_lines = [f"{status_code} {status_msg}"]
            for key, value in headers.items():
                header_lines.append(f"{key}: {value}")
            header_lines.append("")
            
            response_data = "\r\n".join(header_lines).encode("utf-8") + response_body
            
            writer.write(response_data)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _handle_chat_send(self, method, params, body) -> tuple:
        if method != "POST":
            return self._error_response(405, "Method Not Allowed")
        
        try:
            data = json.loads(body)
            session_id = data.get("session_id", "")
            content = data.get("content", "")
            
            if not content:
                return self._error_response(400, "Content is required")
            
            if self._chat_handler:
                result = self._chat_handler.send_message(session_id, "user", content)
                return self._json_response(result)
            return self._error_response(500, "Chat handler not configured")
        except json.JSONDecodeError:
            return self._error_response(400, "Invalid JSON")
        except Exception as e:
            return self._error_response(500, str(e))

    def _handle_chat_history(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        session_id = params.get("session_id", [""])[0]
        limit = int(params.get("limit", ["20"])[0])
        
        if not session_id:
            return self._error_response(400, "Session ID is required")
        
        if self._chat_handler:
            history = self._chat_handler.get_session_history(session_id, limit)
            return self._json_response({"session_id": session_id, "history": history})
        return self._error_response(500, "Chat handler not configured")

    def _handle_chat_sessions(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        if self._chat_handler:
            sessions = self._chat_handler.list_sessions()
            return self._json_response({"sessions": sessions, "count": self._chat_handler.get_session_count()})
        return self._error_response(500, "Chat handler not configured")

    def _handle_chat_create(self, method, params, body) -> tuple:
        if method != "POST":
            return self._error_response(405, "Method Not Allowed")
        
        if self._chat_handler:
            session_id = self._chat_handler.create_session()
            return self._json_response({"session_id": session_id})
        return self._error_response(500, "Chat handler not configured")

    def _handle_tasks_list(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        active_only = params.get("active", ["true"])[0].lower() == "true"
        
        if self._task_tracker:
            if active_only:
                tasks = self._task_tracker.get_active_tasks()
            else:
                tasks = self._task_tracker.get_task_history(50)
            stats = self._task_tracker.get_task_stats()
            return self._json_response({"tasks": tasks, "stats": stats})
        return self._error_response(500, "Task tracker not configured")

    def _handle_tasks_create(self, method, params, body) -> tuple:
        if method != "POST":
            return self._error_response(405, "Method Not Allowed")
        
        try:
            data = json.loads(body)
            task_type = data.get("task_type", "system")
            description = data.get("description", "")
            
            if not description:
                return self._error_response(400, "Description is required")
            
            if self._task_tracker:
                task_id = self._task_tracker.create_task(task_type, description, data.get("metadata"))
                return self._json_response({"task_id": task_id})
            return self._error_response(500, "Task tracker not configured")
        except json.JSONDecodeError:
            return self._error_response(400, "Invalid JSON")

    def _handle_tasks_status(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        task_id = params.get("task_id", [""])[0]
        
        if not task_id:
            return self._error_response(400, "Task ID is required")
        
        if self._task_tracker:
            task = self._task_tracker.get_task(task_id)
            if task:
                return self._json_response(task.to_dict())
            return self._error_response(404, "Task not found")
        return self._error_response(500, "Task tracker not configured")

    def _handle_tasks_update(self, method, params, body) -> tuple:
        if method != "POST":
            return self._error_response(405, "Method Not Allowed")
        
        try:
            data = json.loads(body)
            task_id = data.get("task_id", "")
            
            if not task_id:
                return self._error_response(400, "Task ID is required")
            
            if self._task_tracker:
                action = data.get("action", "")
                if action == "start":
                    self._task_tracker.start_task(task_id)
                elif action == "complete":
                    self._task_tracker.complete_task(task_id, data.get("result"))
                elif action == "fail":
                    self._task_tracker.fail_task(task_id, data.get("error", "Unknown error"))
                elif action == "update":
                    self._task_tracker.update_task(task_id, **data.get("updates", {}))
                
                task = self._task_tracker.get_task(task_id)
                if task:
                    return self._json_response(task.to_dict())
                return self._error_response(404, "Task not found")
            return self._error_response(500, "Task tracker not configured")
        except json.JSONDecodeError:
            return self._error_response(400, "Invalid JSON")

    def _handle_security_overview(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        if self._security_dashboard:
            overview = self._security_dashboard.get_overview()
            return self._json_response(overview)
        return self._error_response(500, "Security dashboard not configured")

    def _handle_security_alerts(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        limit = int(params.get("limit", ["20"])[0])
        
        if self._security_dashboard:
            alerts = self._security_dashboard.get_active_alerts(limit)
            return self._json_response({"alerts": alerts, "count": len(alerts)})
        return self._error_response(500, "Security dashboard not configured")

    def _handle_security_risk(self, method, params, body) -> tuple:
        if method != "POST":
            return self._error_response(405, "Method Not Allowed")
        
        try:
            data = json.loads(body)
            action_description = data.get("action_description", "")
            
            if not action_description:
                return self._error_response(400, "Action description is required")
            
            if self._security_dashboard:
                result = self._security_dashboard.classify_risk(action_description, data.get("context"))
                return self._json_response(result)
            return self._error_response(500, "Security dashboard not configured")
        except json.JSONDecodeError:
            return self._error_response(400, "Invalid JSON")

    def _handle_system_status(self, method, params, body) -> tuple:
        if method != "GET":
            return self._error_response(405, "Method Not Allowed")
        
        status = {
            "server_status": "running" if self._running else "stopped",
            "chat_sessions": self._chat_handler.get_session_count() if self._chat_handler else 0,
            "active_tasks": len(self._task_tracker.get_active_tasks()) if self._task_tracker else 0,
            "websocket_connections": self._websocket_manager.get_connection_count() if self._websocket_manager else 0,
        }
        return self._json_response(status)

    async def _handle_websocket(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        connection_id = f"ws_{uuid.uuid4().hex[:8]}"
        
        if self._websocket_manager:
            self._websocket_manager.register_connection(connection_id, writer)
        
        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(reader.read(8192), timeout=30.0)
                    if not data:
                        break
                    
                    message = data.decode("utf-8", errors="replace")
                    if self._websocket_manager:
                        self._websocket_manager.handle_message(connection_id, message)
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break
        finally:
            if self._websocket_manager:
                self._websocket_manager.unregister_connection(connection_id)

    def _handle_index(self) -> tuple:
        html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGI Agent - Web Interface</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #0f0f23; color: #e0e0e0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .nav { display: flex; gap: 10px; justify-content: center; margin-bottom: 30px; flex-wrap: wrap; }
        .nav button { padding: 10px 20px; border: none; border-radius: 8px; background: #1a1a2e; color: #e0e0e0; cursor: pointer; transition: background 0.3s; }
        .nav button:hover { background: #16213e; }
        .nav button.active { background: #667eea; }
        .section { display: none; }
        .section.active { display: block; }
        .chat-container { max-width: 800px; margin: 0 auto; }
        .chat-history { height: 400px; overflow-y: auto; border: 1px solid #2a2a4a; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #16213e; }
        .chat-message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
        .chat-message.user { background: #0f3460; }
        .chat-message.agent { background: #16213e; border: 1px solid #2a2a4a; }
        .chat-input { display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 12px; border: 1px solid #2a2a4a; border-radius: 8px; background: #16213e; color: #e0e0e0; }
        .chat-input button { padding: 12px 20px; border: none; border-radius: 8px; background: #667eea; color: white; cursor: pointer; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .status-card { background: #16213e; padding: 20px; border-radius: 8px; border: 1px solid #2a2a4a; }
        .status-card h3 { margin-top: 0; color: #667eea; }
        .task-list { list-style: none; padding: 0; }
        .task-item { background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #2a2a4a; }
        .task-progress { height: 5px; background: #2a2a4a; border-radius: 3px; overflow: hidden; margin-top: 10px; }
        .task-progress-bar { height: 100%; background: #667eea; }
        .alert-item { padding: 12px; border-radius: 6px; margin-bottom: 8px; }
        .alert-critical { background: #4a0e0e; border-left: 4px solid #ff0000; }
        .alert-error { background: #4a1a0e; border-left: 4px solid #ff6600; }
        .alert-warning { background: #4a4a0e; border-left: 4px solid #ffff00; }
        .alert-info { background: #0e2a4a; border-left: 4px solid #00aaff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AGI Agent Control Panel</h1>
            <p>Interactive Interface for AGI Agent System</p>
        </div>
        <div class="nav">
            <button class="active" onclick="showSection('chat')">Chat</button>
            <button onclick="showSection('tasks')">Tasks</button>
            <button onclick="showSection('security')">Security</button>
            <button onclick="showSection('system')">System</button>
        </div>
        
        <div id="chat" class="section active">
            <div class="chat-container">
                <div class="chat-history" id="chat-history"></div>
                <div class="chat-input">
                    <input type="text" id="chat-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <div id="tasks" class="section">
            <h2>Task Tracker</h2>
            <div class="status-grid">
                <div class="status-card"><h3>Total Tasks</h3><div id="task-total">0</div></div>
                <div class="status-card"><h3>Active</h3><div id="task-active">0</div></div>
                <div class="status-card"><h3>Completed</h3><div id="task-completed">0</div></div>
                <div class="status-card"><h3>Failed</h3><div id="task-failed">0</div></div>
            </div>
            <ul class="task-list" id="task-list"></ul>
        </div>
        
        <div id="security" class="section">
            <h2>Security Dashboard</h2>
            <div class="status-grid">
                <div class="status-card"><h3>Risk Level</h3><div id="risk-level">unknown</div></div>
                <div class="status-card"><h3>Active Alerts</h3><div id="active-alerts">0</div></div>
                <div class="status-card"><h3>Actions Classified</h3><div id="actions-classified">0</div></div>
                <div class="status-card"><h3>Pending Confirmations</h3><div id="pending-confirmations">0</div></div>
            </div>
            <h3>Active Alerts</h3>
            <div id="alert-list"></div>
        </div>
        
        <div id="system" class="section">
            <h2>System Status</h2>
            <div class="status-grid">
                <div class="status-card"><h3>Server</h3><div id="server-status">unknown</div></div>
                <div class="status-card"><h3>Chat Sessions</h3><div id="chat-sessions">0</div></div>
                <div class="status-card"><h3>Active Tasks</h3><div id="sys-active-tasks">0</div></div>
                <div class="status-card"><h3>WebSocket Connections</h3><div id="ws-connections">0</div></div>
            </div>
        </div>
    </div>
    
    <script>
        let currentSession = '';
        
        function showSection(name) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            document.getElementById(name).classList.add('active');
            event.target.classList.add('active');
            
            if (name === 'tasks') loadTasks();
            if (name === 'security') loadSecurity();
            if (name === 'system') loadSystemStatus();
        }
        
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const content = input.value.trim();
            if (!content) return;
            
            const history = document.getElementById('chat-history');
            history.innerHTML += `<div class="chat-message user"><strong>You:</strong> ${content}</div>`;
            input.value = '';
            history.scrollTop = history.scrollHeight;
            
            try {
                const response = await fetch('/api/chat/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSession, content: content })
                });
                const data = await response.json();
                currentSession = data.session_id;
                
                history.innerHTML += `<div class="chat-message agent"><strong>Agent:</strong> ${data.agent_response?.content || 'No response'}</div>`;
                history.scrollTop = history.scrollHeight;
            } catch (e) {
                history.innerHTML += `<div class="chat-message agent"><strong>Error:</strong> ${e.message}</div>`;
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }
        
        async function loadTasks() {
            try {
                const response = await fetch('/api/tasks/list?active=true');
                const data = await response.json();
                
                document.getElementById('task-total').textContent = data.stats.total_tasks;
                document.getElementById('task-active').textContent = data.stats.active_tasks;
                document.getElementById('task-completed').textContent = data.stats.completed_tasks;
                document.getElementById('task-failed').textContent = data.stats.failed_tasks;
                
                const list = document.getElementById('task-list');
                list.innerHTML = '';
                data.tasks.forEach(task => {
                    list.innerHTML += `
                        <li class="task-item">
                            <strong>${task.description}</strong>
                            <div>Type: ${task.task_type} | Status: ${task.status}</div>
                            <div class="task-progress"><div class="task-progress-bar" style="width: ${task.progress * 100}%"></div></div>
                        </li>
                    `;
                });
            } catch (e) {
                console.error('Failed to load tasks:', e);
            }
        }
        
        async function loadSecurity() {
            try {
                const response = await fetch('/api/security/overview');
                const data = await response.json();
                
                document.getElementById('risk-level').textContent = data.current_risk_level;
                document.getElementById('active-alerts').textContent = data.active_alerts_count;
                document.getElementById('actions-classified').textContent = data.total_actions_classified;
                document.getElementById('pending-confirmations').textContent = data.pending_confirmations;
                
                const list = document.getElementById('alert-list');
                list.innerHTML = '';
                data.active_alerts.forEach(alert => {
                    const severityClass = `alert-${alert.severity}`;
                    list.innerHTML += `<div class="alert-item ${severityClass}">${alert.alert_type}: ${JSON.stringify(alert.payload)}</div>`;
                });
            } catch (e) {
                console.error('Failed to load security:', e);
            }
        }
        
        async function loadSystemStatus() {
            try {
                const response = await fetch('/api/system/status');
                const data = await response.json();
                
                document.getElementById('server-status').textContent = data.server_status;
                document.getElementById('chat-sessions').textContent = data.chat_sessions;
                document.getElementById('sys-active-tasks').textContent = data.active_tasks;
                document.getElementById('ws-connections').textContent = data.websocket_connections;
            } catch (e) {
                console.error('Failed to load system status:', e);
            }
        }
        
        setInterval(() => {
            if (document.getElementById('tasks').classList.contains('active')) loadTasks();
            if (document.getElementById('security').classList.contains('active')) loadSecurity();
            if (document.getElementById('system').classList.contains('active')) loadSystemStatus();
        }, 5000);
    </script>
</body>
</html>
"""
        return (200, "OK", {"Content-Type": "text/html", "Content-Length": len(html)}, html.encode("utf-8"))

    def _handle_not_found(self) -> tuple:
        return self._error_response(404, "Not Found")

    def _json_response(self, data: Dict[str, Any]) -> tuple:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return (200, "OK", {
            "Content-Type": "application/json",
            "Content-Length": len(body),
            "Access-Control-Allow-Origin": "*",
        }, body)

    def _error_response(self, status_code: int, message: str) -> tuple:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        return (status_code, "Error", {
            "Content-Type": "application/json",
            "Content-Length": len(body),
        }, body)

    def stop(self):
        self._running = False
        logger.info("AGI Agent Server stopped")


def run_server(host: str = "0.0.0.0", port: int = 8080) -> AGIAgentServer:
    server = AGIAgentServer(host, port)
    server.run()
    return server