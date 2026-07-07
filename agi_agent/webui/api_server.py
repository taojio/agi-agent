"""
webui/api_server.py - AGI Agent WebUI API 服务器

提供记忆管理、SOUL编辑、任务看板、进化监控等 RESTful API 端点
"""
import os
import sys
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agi_agent.agent import SelfEvolvingAGI
from agi_agent.memory import MemoryTier, MemoryCategory
from agi_agent.soul import SOULParser
from agi_agent.task_engine import TaskPriority

app = FastAPI(title="AGI Agent WebUI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8090", "http://127.0.0.1:8090", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

agi_agent = None


def get_agent() -> SelfEvolvingAGI:
    global agi_agent
    if agi_agent is None:
        agi_agent = SelfEvolvingAGI(input_dim=16)
    return agi_agent


@app.get("/api/agent/info")
async def get_agent_info():
    """获取代理基本信息"""
    agent = get_agent()
    return {
        "name": agent.soul.identity.name,
        "version": agent.soul.version.version,
        "step": agent.train_step,
        "status": "running" if agent.running else "stopped"
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
    return agent.memory_harness.get_all_stats()


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
    
    memories = agent.memory_store.get_tier_store(memory_tier).get_all()
    return {
        "tier": tier,
        "count": len(memories),
        "memories": [m.to_dict() for m in memories[:limit]]
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


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "agi_agent_webui"}


webui_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(webui_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

from fastapi.responses import HTMLResponse, Response

@app.get("/favicon.ico", response_class=Response)
async def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
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
