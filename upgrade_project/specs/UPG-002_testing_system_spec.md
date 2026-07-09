# UPG-002 测试体系建设规格文档

> 模块ID：UPG-002  
> 优先级：P0（最高）  
> 预计周期：3周  
> 版本：v1.0  

---

## 1. 升级目标

将测试覆盖率从 ~30% 提升至 85% 以上，建立完整的自动化测试体系，包括单元测试、集成测试、端到端测试。

### 1.1 核心目标

| 指标 | 当前 | 目标 |
|------|------|------|
| 测试用例数 | 94 | 300+ |
| 整体覆盖率 | ~30% | >= 85% |
| 核心模块覆盖率 | ~50% | >= 90% |
| 安全模块覆盖率 | 0% | >= 95% |
| API 端点覆盖率 | ~20% | >= 90% |
| CI 自动化 | 无 | GitHub Actions |
| 测试类型 | 仅单元 | 单元+集成+端到端+性能 |

### 1.2 测试金字塔

```
        /\
       /  \      E2E 测试 (5-10%)
      /----\
     /      \    集成测试 (20-30%)
    /--------\
   /          \   单元测试 (60-70%)
  /==============\
```

---

## 2. 测试架构

### 2.1 测试框架选型

| 层级 | 框架 | 用途 |
|------|------|------|
| 单元测试 | pytest | 模块级功能测试 |
| 集成测试 | pytest + httpx | API 端点测试 |
| 性能测试 | pytest-benchmark | 性能基准测试 |
| 覆盖率 | pytest-cov | 代码覆盖率统计 |
| Mock | unittest.mock | 依赖模拟 |
| 参数化 | pytest.mark.parametrize | 数据驱动测试 |

### 2.2 目录结构

```
agi_agent/tests/
├── conftest.py              # pytest 配置与 fixtures
├── pytest.ini               # pytest 配置文件
├── unit/                    # 单元测试
│   ├── test_security/       # 安全模块测试
│   │   ├── test_jwt_auth.py
│   │   ├── test_rbac.py
│   │   ├── test_validation.py
│   │   ├── test_rate_limiter.py
│   │   └── test_audit_logger.py
│   ├── test_memory/         # 记忆系统测试
│   │   ├── test_memory_store.py
│   │   └── test_memory_harness.py
│   ├── test_cognitive/      # 认知系统测试
│   │   ├── test_module_synaptic_bus.py
│   │   └── test_dual_system.py
│   ├── test_evolution/      # 进化系统测试
│   ├── test_soul/           # SOUL 系统测试
│   ├── test_task_engine/    # 任务引擎测试
│   └── test_plugins/        # 插件系统测试
├── integration/             # 集成测试
│   ├── test_api_auth.py     # 认证 API 测试
│   ├── test_api_memory.py   # 记忆 API 测试
│   ├── test_api_soul.py     # SOUL API 测试
│   └── test_api_security.py # 安全 API 测试
├── e2e/                     # 端到端测试
│   └── test_agent_flow.py   # Agent 完整流程
└── performance/             # 性能测试
    └── test_performance.py  # 基准性能测试
```

---

## 3. 详细设计

### 3.1 Fixtures 设计

```python
# conftest.py 核心 fixtures

@pytest.fixture
def security_store(tmp_path):
    """临时安全存储"""
    ...

@pytest.fixture
def jwt_auth(security_store):
    """JWT 认证实例"""
    ...

@pytest.fixture
def test_user(jwt_auth):
    """测试用户"""
    ...

@pytest.fixture
def admin_user(jwt_auth):
    """管理员用户"""
    ...

@pytest.fixture
def valid_token(jwt_auth, test_user):
    """有效 access token"""
    ...

@pytest.fixture
def client():
    """FastAPI TestClient"""
    ...

@pytest.fixture
def auth_headers(valid_token):
    """认证请求头"""
    ...
```

### 3.2 安全模块测试用例

| 测试文件 | 用例数 | 覆盖内容 |
|----------|--------|----------|
| test_jwt_auth.py | 20+ | 注册、登录、token验证、刷新、登出、改密 |
| test_rbac.py | 15+ | 角色权限、权限检查、装饰器 |
| test_validation.py | 25+ | 邮箱/密码/用户名/URL/IP/UUID验证、XSS/SQL注入检测 |
| test_rate_limiter.py | 10+ | 限流、超限、重置、清理 |
| test_audit_logger.py | 10+ | 日志记录、查询、统计 |
| test_headers.py | 5+ | 安全头部生成 |

### 3.3 API 集成测试用例

| 测试文件 | 端点 | 用例数 |
|----------|------|--------|
| test_api_auth.py | 登录/注册/刷新/登出/me/改密/角色 | 30+ |
| test_api_memory.py | 列表/添加/搜索/统计 | 15+ |
| test_api_soul.py | 信息/更新/导入/导出 | 10+ |
| test_api_security.py | 审计日志/安全概览 | 10+ |
| test_api_tasks.py | 列表/提交/DAG/统计 | 10+ |

### 3.4 覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| security/ | >= 95% |
| memory/ | >= 90% |
| soul/ | >= 90% |
| task_engine/ | >= 85% |
| cognitive/ | >= 80% |
| evolution/ | >= 75% |
| plugins/ | >= 85% |
| 整体 | >= 85% |

---

## 4. 实施步骤

### 阶段一：测试框架搭建（第1周）
- [ ] 安装 pytest, pytest-cov, httpx
- [ ] 配置 pytest.ini 和 conftest.py
- [ ] 重构测试目录结构
- [ ] 编写基础 fixtures
- [ ] 配置覆盖率报告

### 阶段二：安全模块测试（第1-2周）
- [ ] test_jwt_auth.py - 20+ 用例
- [ ] test_rbac.py - 15+ 用例
- [ ] test_validation.py - 25+ 用例
- [ ] test_rate_limiter.py - 10+ 用例
- [ ] test_audit_logger.py - 10+ 用例
- [ ] 安全模块覆盖率 >= 95%

### 阶段三：核心模块测试（第2周）
- [ ] test_memory_store.py - 20+ 用例
- [ ] test_soul.py - 15+ 用例
- [ ] test_task_engine.py - 15+ 用例
- [ ] test_plugins.py - 10+ 用例

### 阶段四：API 集成测试（第2-3周）
- [ ] test_api_auth.py - 30+ 用例
- [ ] test_api_memory.py - 15+ 用例
- [ ] test_api_soul.py - 10+ 用例
- [ ] test_api_security.py - 10+ 用例
- [ ] API 端点覆盖率 >= 90%

### 阶段五：CI/CD 配置（第3周）
- [ ] GitHub Actions 工作流
- [ ] 自动运行测试
- [ ] 覆盖率报告上传
- [ ] 提交检查门禁
- [ ] 文档更新

---

## 5. 验证标准

- [ ] 单元测试用例 >= 200
- [ ] 集成测试用例 >= 80
- [ ] 整体覆盖率 >= 85%
- [ ] 核心模块覆盖率 >= 90%
- [ ] 安全模块覆盖率 >= 95%
- [ ] 所有测试通过
- [ ] CI 流水线正常运行
- [ ] 覆盖率报告可生成
