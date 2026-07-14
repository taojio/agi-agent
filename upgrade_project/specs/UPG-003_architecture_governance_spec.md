# UPG-003 架构重组与治理规格文档

## 1. 升级目标

### 1.1 现状问题
- 目录命名不一致（meta_cognitive vs metacognition）
- 模块职责重叠（cognitive vs deliberative）
- 导入路径混乱（大量 sys.path.insert）
- 缺乏统一的模块接口规范
- 错误处理不统一
- 配置管理分散

### 1.2 升级目标
- 统一架构分层与命名规范
- 建立核心抽象层（BaseModule, ModuleInterface）
- 标准化异常体系
- 统一配置管理
- 建立依赖注入机制
- 提升代码可维护性与可测试性

## 2. 架构设计

### 2.1 分层架构
```
agi_agent/
├── core/                    # 核心抽象层
│   ├── __init__.py
│   ├── base_module.py      # 模块基类
│   ├── exceptions.py       # 统一异常基类
│   ├── config.py           # 配置管理
│   ├── registry.py         # 模块注册中心
│   └── interfaces.py       # 模块接口定义
├── infrastructure/          # 基础设施层
│   ├── logging.py
│   ├── metrics.py
│   └── cache.py
├── domain/                  # 领域层（各功能模块）
│   ├── perception/
│   ├── cognition/
│   ├── memory/
│   └── ...
├── application/             # 应用层
│   ├── agent.py            # Agent 编排
│   └── services/
└── interfaces/              # 接口层
    ├── webui/
    └── api/
```

### 2.2 模块基类设计
```python
class BaseModule:
    name: str
    version: str
    dependencies: List[str]

    def initialize(self, config: dict) -> None
    def shutdown(self) -> None
    def get_status(self) -> dict
    def health_check(self) -> bool
```

## 3. 实施步骤

### Phase 1: 核心抽象层
- [ ] 创建 core 模块
- [ ] 实现 BaseModule 基类
- [ ] 统一异常体系
- [ ] 配置管理中心

### Phase 2: 基础设施层
- [ ] 统一日志系统
- [ ] 指标收集规范
- [ ] 缓存抽象层

### Phase 3: 模块迁移
- [ ] 安全模块迁移到核心架构
- [ ] 记忆模块迁移
- [ ] 其他模块逐步迁移

## 4. 验证标准
- 所有模块继承 BaseModule
- 统一的异常处理机制
- 配置集中管理
- 模块健康检查通过
