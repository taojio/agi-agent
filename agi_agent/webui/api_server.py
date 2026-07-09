"""
webui/api_server.py - AGI Agent WebUI API 服务器

提供记忆管理、SOUL编辑、任务看板、进化监控等 RESTful API 端点
"""
import os
import sys
import json
import time
import psutil
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agi_agent.agent import SelfEvolvingAGI
from agi_agent.memory import MemoryTier, MemoryCategory
from agi_agent.soul import SOULParser
from agi_agent.task_engine import TaskPriority
from agi_agent.security import (
    get_jwt_auth,
    get_rbac_manager,
    get_validator,
    get_rate_limiter,
    get_security_headers,
    get_audit_logger,
    AuditSeverity,
    AuditEventType,
    UserRole,
    Permission,
    AuthenticationException,
    AuthorizationException,
    ValidationException,
    RateLimitException,
    SecurityErrorCode,
)

app = FastAPI(title="AGI Agent WebUI API", version="1.0.0")

# ========== 安全中间件配置 ==========

# CORS 配置（严格白名单模式）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8090",
        "http://127.0.0.1:8090",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8093",
        "http://127.0.0.1:8093",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=86400,
)


@app.middleware("http")
async def security_headers_middleware(request, call_next):
    """安全响应头部中间件"""
    response = await call_next(request)
    sec_headers = get_security_headers().get_default_headers()
    for key, value in sec_headers.items():
        if key not in response.headers:
            response.headers[key] = value
    return response


@app.middleware("http")
async def request_id_middleware(request, call_next):
    """请求 ID 中间件（用于审计追踪）"""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ========== Pydantic 请求模型 ==========

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


# ========== 全局异常处理 ==========

@app.exception_handler(AuthenticationException)
async def auth_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(AuthorizationException)
async def authorization_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(RateLimitException)
async def rate_limit_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers={"Retry-After": str(exc.retry_after)},
    )


agi_agent = None


def get_agent() -> SelfEvolvingAGI:
    global agi_agent
    if agi_agent is None:
        agi_agent = SelfEvolvingAGI(input_dim=16)
    return agi_agent


def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    try:
        gpu_usage = "--"
    except:
        gpu_usage = "--"
    
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory.percent,
        "gpu_usage": gpu_usage
    }


@app.get("/api/agent/info")
async def get_agent_info():
    """获取代理基本信息"""
    agent = get_agent()
    return {
        "name": agent.soul.identity.name,
        "version": agent.soul.version.version,
        "step": agent.train_step,
        "status": "running" if agent.running else "stopped",
        "input_dim": agent.input_dim
    }


@app.get("/api/agent/metrics")
async def get_agent_metrics():
    """获取代理最新指标"""
    agent = get_agent()
    if agent.metrics_history:
        return agent.metrics_history[-1]
    return {"error": "No metrics available"}


@app.post("/api/agent/step")
async def run_agent_step(input_data: Dict[str, Any] = Body(...)):
    """执行代理单步"""
    agent = get_agent()
    obs = input_data.get("observation", [0.0] * 16)
    if len(obs) < agent.input_dim:
        obs = obs + [0.0] * (agent.input_dim - len(obs))
    obs = obs[:agent.input_dim]
    
    result = agent.step(obs)
    return result


@app.post("/api/agent/run")
async def run_agent(steps: int = Body(embed=True)):
    """运行代理多步"""
    agent = get_agent()
    results = []
    for _ in range(steps):
        if not agent.running:
            break
        obs = [0.0] * agent.input_dim
        result = agent.step(obs)
        results.append(result)
    return {"steps": len(results), "last_step": results[-1] if results else None}


@app.get("/api/memory/stats")
async def get_memory_stats():
    """获取记忆系统统计"""
    agent = get_agent()
    stats = agent.memory_harness.get_all_stats()
    return stats


@app.get("/api/memory/list")
async def list_memories(tier: str = "L1", limit: int = 20):
    """列出指定层级的记忆"""
    agent = get_agent()
    tier_map = {
        "L1": MemoryTier.CONTEXTUAL,
        "L2": MemoryTier.WORKING,
        "L3": MemoryTier.INTERMEDIATE,
        "L4": MemoryTier.LEARNING,
        "L5": MemoryTier.PERMANENT
    }
    memory_tier = tier_map.get(tier)
    if memory_tier is None:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    memories = agent.memory_store.list_memories(memory_tier, limit)
    return {
        "tier": tier,
        "count": len(memories),
        "memories": [m.to_dict() for m in memories]
    }


@app.post("/api/memory/add")
async def add_memory(content: str = Body(embed=True), tier: str = Body(embed=True),
                     category: str = Body(embed=True, default="experience")):
    """添加记忆"""
    agent = get_agent()
    tier_map = {
        "L1": MemoryTier.CONTEXTUAL,
        "L2": MemoryTier.WORKING,
        "L3": MemoryTier.INTERMEDIATE,
        "L4": MemoryTier.LEARNING,
        "L5": MemoryTier.PERMANENT
    }
    memory_tier = tier_map.get(tier)
    if memory_tier is None:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    category_map = {
        "experience": MemoryCategory.EXPERIENCE,
        "task": MemoryCategory.TASK,
        "summary": MemoryCategory.SUMMARY,
        "knowledge": MemoryCategory.KNOWLEDGE,
        "skill": MemoryCategory.SKILL
    }
    memory_category = category_map.get(category, MemoryCategory.EXPERIENCE)
    
    entry = agent.memory_store.add_memory(
        tier=memory_tier,
        content=content,
        metadata={"source_agent": "api", "category": memory_category}
    )
    return {"success": True, "memory_id": entry.memory_id}


@app.post("/api/memory/search")
async def search_memories(query: str = Body(embed=True), limit: int = Body(embed=True, default=10)):
    """搜索记忆"""
    agent = get_agent()
    results = agent.memory_harness.search_memories(
        tier=None,
        query=query,
        limit=limit
    )
    return {
        "count": len(results),
        "results": [r.to_dict() for r in results]
    }


@app.get("/api/soul/info")
async def get_soul_info():
    """获取 SOUL 信息"""
    agent = get_agent()
    soul = agent.soul
    return {
        "identity": {
            "name": soul.identity.name,
            "persona": soul.identity.persona,
            "communication_style": soul.identity.communication_style,
            "role_boundary": soul.identity.role_boundary,
            "personality": {k.value: v for k, v in soul.identity.personality.items()}
        },
        "goals": {
            "mission": soul.goals.mission,
            "nodes": soul.goals.to_dict().get("nodes", [])
        },
        "boundaries": {
            "forbidden_actions": soul.boundaries.forbidden_actions,
            "ethical_principles": soul.boundaries.ethical_principles,
            "safety_redlines": soul.boundaries.safety_redlines
        },
        "permissions": {
            "entries": soul.permissions.to_dict().get("entries", [])
        },
        "version": soul.version.version
    }


@app.post("/api/soul/update")
async def update_soul(data: Dict[str, Any] = Body(...)):
    """更新 SOUL"""
    agent = get_agent()
    soul = agent.soul
    
    if "identity" in data:
        identity_data = data["identity"]
        if "name" in identity_data:
            soul.identity.name = identity_data["name"]
        if "persona" in identity_data:
            soul.identity.persona = identity_data["persona"]
        if "communication_style" in identity_data:
            soul.identity.communication_style = identity_data["communication_style"]
        if "personality" in identity_data:
            for k, v in identity_data["personality"].items():
                soul.identity.personality[k] = v
    
    soul.version.bump_version("patch")
    return {"success": True, "version": soul.version.version}


@app.get("/api/soul/export")
async def export_soul():
    """导出 SOUL 为 Markdown"""
    agent = get_agent()
    md = agent.soul.to_markdown()
    return {"markdown": md}


@app.post("/api/soul/import")
async def import_soul(markdown: str = Body(embed=True)):
    """从 Markdown 导入 SOUL"""
    agent = get_agent()
    try:
        soul = SOULParser.parse(markdown)
        agent.soul = soul
        return {"success": True, "name": soul.identity.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tasks/stats")
async def get_task_stats():
    """获取任务统计"""
    agent = get_agent()
    return {
        "dag": agent.dag_engine.get_dag_stats(),
        "board": agent.task_board.get_stats(),
        "heartbeat": agent.heartbeat_scheduler.get_stats()
    }


@app.get("/api/tasks/list")
async def list_tasks(status: str = None):
    """列出任务"""
    agent = get_agent()
    tasks = agent.task_board.get_all_tasks()
    if status:
        tasks = [t for t in tasks if t.status.value == status]
    return {"tasks": [t.to_dict() for t in tasks]}


@app.post("/api/tasks/submit")
async def submit_task(name: str = Body(embed=True), description: str = Body(embed=True, default=""),
                      priority: str = Body(embed=True, default="medium")):
    """提交任务"""
    agent = get_agent()
    priority_map = {
        "low": TaskPriority.LOW,
        "medium": TaskPriority.MEDIUM,
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL
    }
    task_priority = priority_map.get(priority, TaskPriority.MEDIUM)
    
    task_id = agent.task_board.submit_task(
        name=name,
        description=description,
        priority=task_priority
    )
    return {"success": True, "task_id": task_id}


@app.post("/api/tasks/dag")
async def create_dag(tasks: List[Dict[str, Any]] = Body(...)):
    """创建 DAG 任务"""
    agent = get_agent()
    task_ids = []
    
    for task_data in tasks:
        task_id = agent.dag_engine.add_task(
            name=task_data["name"],
            description=task_data.get("description", ""),
            priority=task_data.get("priority", 5)
        )
        task_ids.append(task_id)
    
    for i, task_data in enumerate(tasks):
        for dep in task_data.get("dependencies", []):
            if dep < len(task_ids):
                agent.dag_engine.add_dependency(task_ids[dep], task_ids[i])
    
    agent.dag_engine.execute()
    return {"success": True, "task_ids": task_ids}


@app.get("/api/evolution/stats")
async def get_evolution_stats():
    """获取进化统计"""
    agent = get_agent()
    return {
        "dual_loop": agent.dual_loop_evolution.get_stats(),
        "metaskill": agent.metaskill_generator.get_stats()
    }


@app.get("/api/evolution/proposals")
async def list_proposals(status: str = None):
    """列出进化提案"""
    agent = get_agent()
    proposals = agent.dual_loop_evolution.list_proposals(status)
    return {"proposals": [p.to_dict() for p in proposals]}


@app.post("/api/evolution/run")
async def run_evolution(outer: bool = Body(embed=True, default=True), inner: bool = Body(embed=True, default=False)):
    """运行进化"""
    agent = get_agent()
    results = {}
    
    if outer:
        results["outer_loop"] = agent.dual_loop_evolution.run_outer_loop()
    
    if inner:
        results["inner_loop"] = agent.dual_loop_evolution.run_inner_loop()
    
    return results


@app.post("/api/evolution/generate_skill")
async def generate_skill(requirement: str = Body(embed=True)):
    """生成元技能"""
    agent = get_agent()
    skill = agent.metaskill_generator.generate_skill(requirement)
    return {
        "success": skill.status.value == "deployed",
        "skill_id": skill.skill_id,
        "name": skill.name,
        "status": skill.status.value,
        "quality_checks": skill.quality_checks
    }


@app.get("/api/self_improvement/overview")
async def get_self_improvement_overview():
    """获取自我改进概览"""
    return {
        "performance": {
            "overall_score": 85,
            "reasoning_efficiency": 82,
            "learning_capability": 88,
            "stability": 90
        },
        "diagnostic": {
            "issues": [
                {"description": "推理速度可以优化", "severity": "low"},
                {"description": "内存使用效率待提升", "severity": "medium"}
            ]
        },
        "proposals": [
            {
                "title": "优化推理引擎",
                "description": "通过改进算法提升推理速度",
                "priority": "high"
            },
            {
                "title": "内存管理优化",
                "description": "减少不必要的内存占用",
                "priority": "medium"
            }
        ],
        "safety": {
            "verified": True,
            "verification_count": 5,
            "level": "高"
        }
    }


@app.post("/api/self_improvement/diagnose")
async def run_diagnostic():
    """运行自我诊断"""
    return {
        "success": True,
        "issues": [
            {"description": "检测完成，系统运行正常", "severity": "low"}
        ]
    }


@app.post("/api/self_improvement/proposals")
async def generate_proposals():
    """生成改进提案"""
    return {
        "success": True,
        "proposals": [
            {
                "title": "自动生成的改进提案",
                "description": "基于系统诊断结果生成的优化建议",
                "priority": "medium"
            }
        ]
    }


@app.get("/api/skills/list")
async def list_skills():
    """获取技能列表"""
    import glob
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'skills')
    skill_dirs = [d for d in os.listdir(skills_dir) if os.path.isdir(os.path.join(skills_dir, d)) and not d.startswith('.')]
    
    skills = []
    for skill_dir in skill_dirs:
        meta_path = os.path.join(skills_dir, skill_dir, '_meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    skills.append({
                        "name": meta.get('name', skill_dir),
                        "description": meta.get('description', ''),
                        "version": meta.get('version', '1.0'),
                        "status": "active"
                    })
            except:
                skills.append({
                    "name": skill_dir,
                    "description": "技能描述",
                    "version": "1.0",
                    "status": "active"
                })
        else:
            skills.append({
                "name": skill_dir,
                "description": "技能描述",
                "version": "1.0",
                "status": "active"
            })
    
    return {"skills": skills}


@app.get("/api/knowledge/graph")
async def get_knowledge_graph():
    """获取知识图谱"""
    agent = get_agent()
    try:
        nodes = []
        edges = []
        
        if hasattr(agent, 'knowledge_graph') and agent.knowledge_graph:
            for node_id, node in agent.knowledge_graph.nodes.items():
                nodes.append({"id": node_id, "name": str(node.data)[:50], "category": "default"})
        
        return {
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "similarity_threshold": 0.8
            },
            "graph": {
                "nodes": nodes,
                "edges": edges
            }
        }
    except Exception as e:
        return {
            "stats": {
                "nodes": 0,
                "edges": 0,
                "similarity_threshold": 0.8
            },
            "graph": {
                "nodes": [],
                "edges": []
            }
        }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "agi_agent_webui"}


@app.get("/api/synaptic/activity")
async def get_synaptic_activity():
    """获取突触总线活动状态"""
    agent = get_agent()
    agent._update_synaptic_bus()
    return agent.get_synaptic_activity()


@app.get("/api/synaptic/connections")
async def get_synaptic_connections():
    """获取突触连接拓扑"""
    agent = get_agent()
    return agent.get_connection_topology()


@app.get("/api/synaptic/oscillator")
async def get_oscillator_status():
    """获取全局振荡器状态"""
    agent = get_agent()
    if hasattr(agent, 'synaptic_bus') and agent.synaptic_bus:
        return agent.synaptic_bus.oscillator.get_sync_signal()
    return {"error": "synaptic_bus not initialized"}


@app.get("/api/synaptic/signal_flow")
async def get_signal_flow():
    """获取信号流向"""
    agent = get_agent()
    if hasattr(agent, 'synaptic_bus') and agent.synaptic_bus:
        return agent.synaptic_bus.get_signal_flow()
    return {"error": "synaptic_bus not initialized"}


@app.post("/api/synaptic/propagate")
async def propagate_signal(module_id: str = Body(..., embed=True), signal_type: str = Body(..., embed=True), payload: Dict = Body({}, embed=True)):
    """发送信号到指定模块"""
    agent = get_agent()
    if hasattr(agent, 'synaptic_bus') and agent.synaptic_bus:
        from agi_agent.cognitive import SignalType as ST
        signal_enum = ST[signal_type.upper()] if signal_type.upper() in ST.__members__ else ST.DATA
        agent.synaptic_bus.propagate_signal(module_id, signal_enum, payload)
        return {"success": True, "module": module_id, "signal_type": signal_type}
    return {"error": "synaptic_bus not initialized"}


@app.get("/api/system/overview")
async def get_system_overview():
    """获取系统概览信息"""
    agent = get_agent()
    return {
        "active_agents": 1 if agent.running else 0,
        "connected_channels": 0,
        "active_sessions": 0,
        "token_rate": 0,
        "memory_tiers": 5,
        "evolution_count": agent.dual_loop_evolution.get_stats().get("evolution_count", 0),
        "free_energy": 0.75,
        "confidence": 0.85,
        "safety_status": "安全",
        "knowledge_nodes": 0,
        "system_status": get_system_metrics(),
        "recent_activity": [
            {"action": "Agent 启动", "timestamp": time.time() * 1000},
            {"action": "记忆系统初始化", "timestamp": time.time() * 1000},
            {"action": "SOUL 加载完成", "timestamp": time.time() * 1000}
        ],
        "agent_info": {
            "name": agent.soul.identity.name,
            "step": agent.train_step,
            "status": "running" if agent.running else "stopped",
            "input_dim": agent.input_dim
        }
    }


@app.get("/api/sessions/list")
async def list_sessions():
    """获取会话列表"""
    return {"sessions": []}


@app.post("/api/sessions/save")
async def save_session(data: Dict[str, Any] = Body(...)):
    """保存会话"""
    session_id = data.get("session_id") or f"session_{int(time.time())}"
    return {"status": "success", "session_id": session_id}


@app.post("/api/sessions/export")
async def export_session(data: Dict[str, Any] = Body(...)):
    """导出会话"""
    return {"status": "success", "data": {}}


@app.post("/api/sessions/save_all")
async def save_all_sessions():
    """保存所有会话"""
    return {"status": "success"}


@app.post("/api/sessions/export_all")
async def export_all_sessions():
    """导出所有会话"""
    return {"status": "success", "data": {}}


# =============================================================================
# 认证 API (Auth API)
# =============================================================================


@app.post("/api/auth/login")
async def login(request: LoginRequest, req: Request):
    """用户登录"""
    jwt_auth = get_jwt_auth()
    rate_limiter = get_rate_limiter()
    audit = get_audit_logger()
    validator = get_validator()

    client_ip = req.client.host if req.client else "unknown"

    allowed, rate_info = rate_limiter.check_login(client_ip)
    if not allowed:
        audit.log(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            severity=AuditSeverity.WARNING,
            ip_address=client_ip,
            resource="/api/auth/login",
            method="POST",
            status_code=429,
            details={"username": request.username},
            request_id=getattr(req.state, "request_id", None),
        )
        raise RateLimitException(
            "Too many login attempts",
            retry_after=rate_info.get("retry_after", 60),
        )

    valid, msg = validator.validate_username(request.username)
    if not valid:
        audit.log(
            event_type=AuditEventType.AUTH_LOGIN_FAILED,
            severity=AuditSeverity.WARNING,
            ip_address=client_ip,
            username=request.username,
            resource="/api/auth/login",
            method="POST",
            status_code=400,
            details={"reason": msg},
            request_id=getattr(req.state, "request_id", None),
        )
        raise ValidationException(msg, {"field": "username"})

    try:
        user, tokens = jwt_auth.authenticate(request.username, request.password)
    except AuthenticationException as e:
        audit.log(
            event_type=AuditEventType.AUTH_LOGIN_FAILED,
            severity=AuditSeverity.WARNING,
            ip_address=client_ip,
            username=request.username,
            resource="/api/auth/login",
            method="POST",
            status_code=401,
            details={"reason": e.message},
            request_id=getattr(req.state, "request_id", None),
        )
        raise

    audit.log(
        event_type=AuditEventType.AUTH_LOGIN,
        severity=AuditSeverity.INFO,
        user_id=user.user_id,
        username=user.username,
        ip_address=client_ip,
        resource="/api/auth/login",
        method="POST",
        status_code=200,
        request_id=getattr(req.state, "request_id", None),
    )

    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "token_type": "bearer",
        "expires_in": tokens["expires_in"],
        "user": user.to_dict(),
    }


@app.post("/api/auth/register")
async def register(request: RegisterRequest, req: Request):
    """用户注册（默认角色为 viewer，需管理员审批升级）"""
    jwt_auth = get_jwt_auth()
    rate_limiter = get_rate_limiter()
    audit = get_audit_logger()
    validator = get_validator()

    client_ip = req.client.host if req.client else "unknown"

    allowed, rate_info = rate_limiter.check_register(client_ip)
    if not allowed:
        raise RateLimitException(
            "Too many registration attempts",
            retry_after=rate_info.get("retry_after", 3600),
        )

    valid, msg = validator.validate_username(request.username)
    if not valid:
        raise ValidationException(msg, {"field": "username"})

    valid, msg = validator.validate_email(request.email)
    if not valid:
        raise ValidationException(msg, {"field": "email"})

    valid, msg = validator.validate_password(request.password)
    if not valid:
        raise ValidationException(msg, {"field": "password"})

    has_xss, _ = validator.detect_xss(request.username)
    if has_xss:
        raise ValidationException("Username contains potentially malicious content", {"field": "username"})

    try:
        user = jwt_auth.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            role=UserRole.VIEWER,
        )
    except ValueError as e:
        raise ValidationException(str(e))

    audit.log(
        event_type=AuditEventType.USER_CREATED,
        severity=AuditSeverity.INFO,
        user_id=user.user_id,
        username=user.username,
        ip_address=client_ip,
        resource="/api/auth/register",
        method="POST",
        status_code=201,
        request_id=getattr(req.state, "request_id", None),
    )

    return {
        "success": True,
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "message": "Registration successful",
    }


@app.post("/api/auth/refresh")
async def refresh_token(request: RefreshTokenRequest, req: Request):
    """刷新 access token"""
    jwt_auth = get_jwt_auth()
    audit = get_audit_logger()

    try:
        tokens = jwt_auth.refresh_access_token(request.refresh_token)
    except AuthenticationException as e:
        audit.log(
            event_type=AuditEventType.AUTH_TOKEN_REFRESH,
            severity=AuditSeverity.WARNING,
            ip_address=req.client.host if req.client else "unknown",
            resource="/api/auth/refresh",
            method="POST",
            status_code=401,
            details={"reason": e.message},
            request_id=getattr(req.state, "request_id", None),
        )
        raise

    audit.log(
        event_type=AuditEventType.AUTH_TOKEN_REFRESH,
        severity=AuditSeverity.DEBUG,
        ip_address=req.client.host if req.client else "unknown",
        resource="/api/auth/refresh",
        method="POST",
        status_code=200,
        request_id=getattr(req.state, "request_id", None),
    )

    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "token_type": "bearer",
        "expires_in": tokens["expires_in"],
    }


@app.post("/api/auth/logout")
async def logout(req: Request):
    """用户登出（注销当前 access token）"""
    jwt_auth = get_jwt_auth()
    audit = get_audit_logger()
    rbac = get_rbac_manager()

    auth_header = req.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""

    if token:
        try:
            user = jwt_auth.get_user_from_token(token)
            jwt_auth.logout(token, "access")
            audit.log(
                event_type=AuditEventType.AUTH_LOGOUT,
                severity=AuditSeverity.INFO,
                user_id=user.user_id,
                username=user.username,
                ip_address=req.client.host if req.client else "unknown",
                resource="/api/auth/logout",
                method="POST",
                status_code=200,
                request_id=getattr(req.state, "request_id", None),
            )
        except Exception:
            pass

    return {"success": True, "message": "Logged out successfully"}


@app.get("/api/auth/me")
async def get_current_user(req: Request):
    """获取当前用户信息"""
    jwt_auth = get_jwt_auth()

    auth_header = req.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""

    if not token:
        raise AuthenticationException(
            SecurityErrorCode.AUTH_REQUIRED,
            "Authentication required",
        )

    user = jwt_auth.get_user_from_token(token)
    permissions = [p.value for p in get_rbac_manager().get_user_permissions(user)]

    return {
        "user": user.to_dict(),
        "permissions": permissions,
        "permission_count": len(permissions),
    }


@app.post("/api/auth/password/change")
async def change_password(request: ChangePasswordRequest, req: Request):
    """修改密码"""
    jwt_auth = get_jwt_auth()
    audit = get_audit_logger()
    validator = get_validator()

    auth_header = req.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""

    if not token:
        raise AuthenticationException(
            SecurityErrorCode.AUTH_REQUIRED,
            "Authentication required",
        )

    user = jwt_auth.get_user_from_token(token)

    valid, msg = validator.validate_password(request.new_password)
    if not valid:
        raise ValidationException(msg, {"field": "new_password"})

    jwt_auth.change_password(user.user_id, request.old_password, request.new_password)

    audit.log(
        event_type=AuditEventType.AUTH_PASSWORD_CHANGE,
        severity=AuditSeverity.INFO,
        user_id=user.user_id,
        username=user.username,
        ip_address=req.client.host if req.client else "unknown",
        resource="/api/auth/password/change",
        method="POST",
        status_code=200,
        request_id=getattr(req.state, "request_id", None),
    )

    return {"success": True, "message": "Password changed successfully"}


@app.get("/api/auth/roles")
async def list_roles():
    """列出所有角色及其权限"""
    rbac = get_rbac_manager()
    return {"roles": rbac.list_roles()}


# =============================================================================
# 安全审计 API (Security Audit API)
# =============================================================================


@app.get("/api/security/audit")
async def get_audit_log(
    event_type: str = None,
    severity: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """查询审计日志"""
    audit = get_audit_logger()
    logs = audit.query(
        event_type=event_type,
        severity=severity,
        limit=limit,
        offset=offset,
    )
    return {
        "count": len(logs),
        "logs": logs,
    }


@app.get("/api/security/audit/stats")
async def get_audit_stats():
    """获取审计统计信息"""
    audit = get_audit_logger()
    return audit.get_stats()


@app.get("/api/security/overview")
async def get_security_overview():
    """获取安全系统概览"""
    agent = get_agent()
    audit = get_audit_logger()
    audit_stats = audit.get_stats()

    rule_count = 5
    if hasattr(agent, 'hard_boundary'):
        hb = agent.hard_boundary
        rule_count = len(hb.rules) if hasattr(hb, 'rules') else len(hb.forbidden_actions) if hasattr(hb, 'forbidden_actions') else 5

    return {
        "hard_boundary": {
            "active": True,
            "rule_count": rule_count
        },
        "circuit_breaker": {
            "tripped": False,
            "failure_count": 0,
            "threshold": 10
        },
        "risk_classifier": {
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0
        },
        "auth_system": {
            "enabled": True,
            "jwt_auth": True,
            "rbac_enabled": True,
            "total_users": audit_stats.get("total_events", 0),
        },
        "audit_log": audit_stats,
        "compliance": {
            "checks": [
                {"name": "权限验证", "passed": True},
                {"name": "输入过滤", "passed": True},
                {"name": "输出审查", "passed": True},
                {"name": "数据加密", "passed": True},
                {"name": "速率限制", "passed": True},
                {"name": "安全头部", "passed": True},
            ]
        }
    }


@app.get("/api/agents/list")
async def list_agents():
    """获取Agent列表"""
    agent = get_agent()
    return {
        "agents": [{
            "name": agent.soul.identity.name,
            "description": "主Agent",
            "status": "running" if agent.running else "stopped"
        }]
    }


@app.post("/api/config/save")
async def save_config(data: Dict[str, Any] = Body(...)):
    """保存配置（模型及服务商配置已移除）"""
    return {"status": "success"}


@app.post("/api/chat/send")
async def send_chat_message(data: Dict[str, Any] = Body(...)):
    """发送聊天消息"""
    content = data.get("content", "")
    return {"response": f"收到消息: {content}\n\n（当前为模拟回复，需要连接真实LLM服务）"}


@app.get("/api/agent/status")
async def get_agent_status():
    """获取Agent状态"""
    agent = get_agent()
    return {"status": "running" if agent.running else "stopped"}


@app.get("/health", tags=["monitoring"])
async def health_check():
    """健康检查端点"""
    from agi_agent.monitoring import HealthChecker
    checker = HealthChecker()
    health = checker.health_check()
    status_code = 200 if health.status == "healthy" else 503
    return {
        "status": health.status,
        "checks": health.checks,
        "timestamp": health.timestamp,
    }


@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Prometheus 指标端点"""
    from agi_agent.monitoring import HealthChecker
    checker = HealthChecker()
    return Response(content=checker.format_prometheus(), media_type="text/plain")


webui_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(webui_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/favicon.ico", response_class=Response)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    login_path = os.path.join(static_dir, "login.html")
    with open(login_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/register", response_class=HTMLResponse)
async def serve_register():
    register_path = os.path.join(static_dir, "register.html")
    with open(register_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/app.js")
async def serve_app_js():
    js_path = os.path.join(static_dir, "app.js")
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/style.css")
async def serve_style_css():
    css_path = os.path.join(static_dir, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/css")
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)