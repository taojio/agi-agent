# AGI Agent - Self-Evolving Autonomous Intelligence

🤖 一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统

## 项目简介 / Project Description

AGI Agent 是一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统。它采用先进的认知架构，能够持续学习、自适应进化，并在复杂环境中做出自主决策。系统内置五级记忆底座、脉冲神经网络（SNN）认知引擎、多智能体协作网络、JWT安全认证体系、神经符号推理系统、世界模型，以及完整的自我改进与进化机制。

---

## 📋 目录 / Table of Contents

- [项目简介 / Project Description](#项目简介--project-description)
- [核心特性 / Core Features](#核心特性--core-features)
- [跨学科知识规则体系 / Knowledge Rulebase](#跨学科知识规则体系--knowledge-rulebase)
- [技术架构 / Technical Architecture](#技术架构--technical-architecture)
- [安装指南 / Installation](#安装指南--installation)
- [使用教程 / Usage Tutorial](#使用教程--usage-tutorial)
- [API 接口 / API Reference](#api-接口--api-reference)
- [开发指南 / Development Guide](#开发指南--development-guide)
- [测试验证 / Testing](#测试验证--testing)
- [安全框架 / Security Framework](#安全框架--security-framework)
- [升级项目 / Project Phoenix](#升级项目--project-phoenix)
- [贡献指南 / Contributing](#贡献指南--contributing)
- [许可证 / License](#许可证--license)

---

## 🌟 核心特性 / Core Features

### 认知系统 / Cognitive System
- **自我意识系统** (Self-Awareness) - 能够识别自身存在、能力边界和局限性
- **自主思考机制** (Autonomous Thinking) - 基于双系统认知的深度推理能力
- **因果推理引擎** (Causal Reasoning) - 多层次因果关系建模与推理
- **预测编码** (Predictive Coding) - 基于自由能原理的主动预测
- **脉冲神经网络** (SNN) - 立体SNN、增强SNN、生物听觉SNN等多型态神经架构
- **模块突触总线** (Module Synaptic Bus) - 模块间突触连接构建类脑数据通路
- **神经符号推理系统** (Neuro-Symbolic Reasoner) - 符号逻辑与神经网络深度融合，支持结构化知识表示、规则推理及可解释性推理
- **世界模型** (World Model) - 对环境的"内部心智模拟"，实现预测、泛化与推理三大核心能力：
  - **多模态感知编码** - 视觉/3D分支、语义/文本分支、传感器分支、统一投影层的深度融合
  - **分层抽象表征** - 感知级→物体级→场景级→规则级四级表征，支持双向信息流动
  - **世界动力学预测引擎** - 微观物理预测器、中观物体预测器、宏观语义预测器及闭环修正机制
  - **因果结构推理** - 因果发现、干预式预测、分布外适应与反事实推理
  - **记忆与知识库** - 工作记忆、情景记忆、语义记忆三级记忆系统
  - **规划与决策交互层** - 想象推理接口、目标导向接口、强化学习接口

### 记忆与学习 / Memory & Learning
- **工业级记忆底座 v2.0** - 分层、持久化、可检索、可遗忘、可溯源、多模态统一
- **五级记忆系统** (L1-L5) - 瞬时/上下文/短期/中期/学习/永久记忆完整层级
- **插件化记忆架构** - 检索、遗忘、巩固、压缩、摘要、聚类、自省插件可替换
- **多模态数据载荷** - 文本、图像、音频、结构化数据、数值指标统一存储范式
- **存储后端抽象层** - 支持内存/SQLite/Qdrant/PostgreSQL/Redis多后端切换
- **记忆安全层** - 权限分级(PUBLIC/INTERNAL/CONFIDENTIAL/RESTRICTED)、敏感数据脱敏
- **知识图谱** (Knowledge Graph) - 结构化知识存储与推理
- **元学习** (Meta-Learning) - 自适应学习策略与超参数优化
- **文件摄入系统** (File Ingestion) - 支持多类型文件的导入、处理与向量化

### 决策与行动 / Decision & Action
- **自主决策引擎** (Autonomous Decision Engine) - 基于风险评估的智能决策
- **目标分解** (Target Decomposition) - 复杂任务的层次化分解
- **路径规划** (Path Planning) - 行动路径的优化规划
- **主动探索** (Active Exploration) - 未知领域的自主探索
- **统一编排层 v2** - 事件总线、任务DAG、编排引擎三位一体

### 进化与自我改进 / Evolution & Self-Improvement
- **四级进化机制** (Quad-Level Evolution) - 微观到宏观的多层次进化
- **递归自我改进** (Recursive Self-Improvement) - 持续自我优化能力
- **架构变异** (Architecture Mutation) - 神经网络结构的自适应调整
- **性能评估** (Performance Evaluation) - 实时性能监控与改进
- **引导式自改进** (Bootstrapped Self-Improver) - 基于黑盒验证的参数优化

### 安全与合规 / Safety & Compliance
- **JWT认证系统** - 双令牌机制(access/refresh)、本地用户管理
- **RBAC权限模型** - 5级角色(SUPER_ADMIN/ADMIN/OPERATOR/VIEWER/GUEST)、22项权限
- **输入验证** - XSS/SQL注入检测、邮箱/用户名/密码/URL/IP/UUID验证
- **速率限制** - 基于Redis滑动窗口算法的IP级和用户级双重限流
- **安全响应头** - CSP/HSTS/X-Content-Type-Options等11项安全头
- **审计日志系统** - 安全事件记录、查询和统计
- **硬边界系统** (Hard Boundary) - 不可逾越的安全底线
- **熔断机制** (Circuit Breaker) - 异常自动熔断保护
- **多层次安全防护** (Multi-layer Safety) - 硬边界、合规检查、风险分类
- **安全监控** (Safety Monitor) - 实时安全约束与紧急停机
- **审计追踪** (Audit Trail) - 完整的决策过程记录

### 多智能体协作 / Multi-Agent Collaboration
- **智能体集群** (Agent Swarm) - 多智能体注册与管理
- **协作模式** - 竞争/合作/混合三种协作模式
- **任务分配器** (Task Allocator) - 智能任务分发
- **共享记忆空间** (Shared Memory) - 智能体间信息共享
- **冲突解决器** (Conflict Resolver) - 多智能体冲突协调

### 部署与运维 / Deployment & Operations
- **健康检查** - `/health` 端点，系统健康状态监控
- **Prometheus指标** - `/metrics` 端点，标准化指标暴露
- **Docker部署** - Dockerfile + docker-compose 一键部署
- **系统托盘** - Windows系统托盘集成
- **资源管理** - CPU/内存/GPU 实时监控

---

## 📚 跨学科知识规则体系 / Knowledge Rulebase

### 概述

系统内置跨学科知识规则体系，包含 **5个学科、121条规则**，支持规则检索、计算和跨学科知识融合。

### 学科分布

| 学科 | 规则数量 | 核心内容 |
|------|----------|----------|
| **物理** | 22条 | 牛顿定律、万有引力、能量守恒、热力学、电磁学、简谐运动、光学、相对论基础 |
| **数学** | 26条 | 代数、几何、微积分、概率统计、三角函数、坐标系、向量、复数 |
| **化学** | 22条 | 原子结构、化学键、化学反应、溶液、元素周期表、化学平衡、电化学、有机化学 |
| **生物** | 23条 | 细胞生物学、遗传学、生理学、生态学、酶与代谢、蛋白质与核酸、细胞器、免疫调节 |
| **语文** | 28条 | 拼音、汉字、词语、成语、句式结构、修辞手法、标点符号、写作技巧 |

### 核心规则示例

**物理**
- 牛顿第二定律: F = m·a
- 万有引力定律: F = G·m₁·m₂/r²
- 质能方程: E = mc²
- 单摆周期: T = 2π·√(L/g)

**数学**
- 勾股定理: a² + b² = c²
- 圆的面积: S = πr²
- 两点距离公式: d = √((x₂-x₁)² + (y₂-y₁)²)
- 欧拉公式: e^(iθ) = cos(θ) + i·sin(θ)

**化学**
- pH值计算: pH = -log[H⁺]
- 摩尔质量: M = m/n
- 元素周期律: 同周期原子半径递减
- 化学平衡常数: K = [C]^c·[D]^d/[A]^a·[B]^b

**生物**
- 光合作用: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
- 中心法则: DNA → RNA → 蛋白质
- 酶的特性: 高效性、专一性、温和条件
- 细胞学说: 所有生物由细胞构成

**语文**
- 声调规则: 一声平 二声扬 三声拐弯 四声降
- 句子成分: 主语 + 谓语 + 宾语 + 定语 + 状语 + 补语
- 修辞手法: 比喻、拟人、夸张、排比
- 标点符号: 句号、逗号、问号、感叹号、引号

### 使用示例

```python
from agi_agent.knowledge_rulebase import (
    DisciplinaryRuleRegistry,
    register_default_disciplines,
    Discipline,
)

registry = DisciplinaryRuleRegistry()
register_default_disciplines(registry)

result = registry.get_rule("physics_newton_2").apply({"m": 10, "a": 5})
print(f"牛顿第二定律结果: {result}")

result = registry.get_rule("math_pythagorean").apply({"a": 3, "b": 4})
print(f"勾股定理结果: {result}")

results = registry.search_by_concept("能量")
for r in results:
    print(f"  - [{r.discipline.value}] {r.name}")
```

### 模块文件

| 文件 | 描述 |
|------|------|
| `knowledge_rulebase/disciplinary_rule.py` | 规则数据模型和公式计算引擎 |
| `knowledge_rulebase/rule_registry.py` | 规则注册中心和检索系统 |
| `knowledge_rulebase/physics_rules.py` | 物理学科规则集 |
| `knowledge_rulebase/math_rules.py` | 数学学科规则集 |
| `knowledge_rulebase/chemistry_rules.py` | 化学学科规则集 |
| `knowledge_rulebase/biology_rules.py` | 生物学科规则集 |
| `knowledge_rulebase/chinese_rules.py` | 语文学科规则集 |
| `knowledge_rulebase/integration.py` | 跨学科知识融合管理器 |

---

## 🏗️ 技术架构 / Technical Architecture

### 系统架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                      SelfEvolvingAGI Agent                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │   感知层     │───►│   认知层     │───►│   执行层     │             │
│   │  Perception │    │  Cognitive  │    │  Execution  │             │
│   │  AutoEncoder│    │  Inference  │    │  Action     │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                  │                       │
│        ▼                  ▼                  ▼                       │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  记忆底座   │    │  决策引擎   │    │  人格系统   │             │
│   │  Memory v2  │    │  Decision   │    │  Personality│             │
│   │  L1-L5+插件 │    │  Engine     │    │  Core       │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                                          │
│        ▼                  ▼                                          │
│   ┌─────────────┐    ┌─────────────┐                                │
│   │  知识规则库  │    │  元认知系统  │                                │
│   │  Rulebase   │    │  Meta-Cog   │                                │
│   │  5学科/121条 │    │  Orchestrator│                               │
│   └─────────────┘    └─────────────┘                                │
│                                                                      │
│   ┌───────────────────────────────────────────────────────┐         │
│   │              神经符号推理 & 世界模型                    │         │
│   │  Neuro-Symbolic Reasoner │     World Model Engine     │         │
│   │  ────────────────────────│─────────────────────────── │         │
│   │  Symbolic Logic          │ ┌──────────────────────┐   │         │
│   │  Neural Integration      │ │ 多模态感知编码模块   │   │         │
│   │  Rule Reasoning          │ │ Visual/Text/Sensor   │   │         │
│   │  Interpretable Reasoning │ │ Unified Projection   │   │         │
│   │                          │ └──────────────────────┘   │         │
│   │                          │           │                 │         │
│   │                          │ ┌──────────────────────┐   │         │
│   │                          │ │ 分层抽象表征模块     │   │         │
│   │                          │ │ Perceptual→Object    │   │         │
│   │                          │ │ Scene→Rule           │   │         │
│   │                          │ └──────────────────────┘   │         │
│   │                          │           │                 │         │
│   │                          │ ┌──────────────────────┐   │         │
│   │                          │ │ 动力学预测引擎(核心) │   │         │
│   │                          │ │ Micro/Meso/Macro     │   │         │
│   │                          │ │ Closed-Loop Correction│   │         │
│   │                          │ └──────────────────────┘   │         │
│   │                          │           │                 │         │
│   │                          │ ┌──────────────────────┐   │         │
│   │                          │ │ 因果结构推理模块     │   │         │
│   │                          │ │ Causal Discovery     │   │         │
│   │                          │ │ Intervention/Counterfactual│      │         │
│   │                          │ └──────────────────────┘   │         │
│   │                          │           │                 │         │
│   │                          │ ┌──────────────────────┐   │         │
│   │                          │ │ 记忆与知识库         │   │         │
│   │                          │ │ Working/Episodic     │   │         │
│   │                          │ │ Semantic Memory      │   │         │
│   │                          │ └──────────────────────┘   │         │
│   │                          │           │                 │         │
│   │                          │ ┌──────────────────────┐   │         │
│   │                          │ │ 规划与决策交互层     │   │         │
│   │                          │ │ Imagination/Goal/RL  │   │         │
│   │                          │ └──────────────────────┘   │         │
│   │  ────────────────────────│─────────────────────────── │         │
│   │           Coordinator & Knowledge Synchronization      │         │
│   └───────────────────────────────────────────────────────┘         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  进化引擎   │    │  自我改进   │    │  安全体系   │             │
│   │  Evolution  │    │  Self-Improve│   │  JWT+RBAC   │             │
│   │  Engine     │    │  Engine     │    │  +Audit     │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  编排层 v2  │    │  多智能体   │    │  监控运维   │             │
│   │  Event Bus  │    │  Agent Swarm│    │  Health+Met │             │
│   │  Task DAG   │    │  Shared Mem │    │  Docker     │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  文件摄入   │    │  插件系统   │    │  技能管理   │             │
│   │  File       │    │  Plugin     │    │  Skills     │             │
│   │  Ingestion  │    │  Manager    │    │  Manager    │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 模块详解

| 模块 | 职责 | 核心组件 | 文件位置 |
|------|------|----------|----------|
| **perception** | 感官输入处理与特征提取 | GrowingAutoEncoder, MultimodalFusion | `perception/` |
| **cognitive** | 时序预测与自主推理 | DualSystemCognition, CausalReasoningEngine, SNN, WorldModelEngine, MultiModalEncoder, HierarchicalRepresentation, DynamicsPredictor, CausalReasoner, MemorySystem, PlanningInterface | `cognitive/` |
| **memory** | 工业级记忆底座 v2.0 | MemoryHarness, MemoryStore, PluginRegistry, StorageBackend | `memory/` |
| **learning** | 元学习与知识管理 | MetaLearningLayer, KnowledgeGraph, KnowledgeIntegrator | `learning/` |
| **decision** | 自主决策与行动规划 | AutonomousDecisionEngine, ActionPlanner, DecisionOptimizer | `decision/` |
| **evolution** | 神经进化与结构优化 | EvolutionEngine (NEAT), QuadLevelEvolution, MetaEvolution | `evolution/`, `meta_evolution/` |
| **meta_cognitive** | 自我监控与资源调度 | MetaCognitiveOrchestrator, SelfModel, ReflectionEngine | `meta_cognitive/` |
| **self_improvement** | 递归自我改进 | RecursiveSelfImprover, BootstrappedSelfImprover, AutomatedImprovementLoop | `self_improvement/` |
| **security** | 安全认证与合规检查 | JWTAuth, RBAC, RateLimiter, AuditLogger, SafetyMonitor, HardBoundary, CircuitBreaker | `security/` |
| **orchestration** | 统一编排层 v2 | EventBus, TaskDAG, Orchestrator | `orchestration/` |
| **multi_agent** | 多智能体协作 | AgentSwarm, TaskAllocator, SharedMemorySpace | `multi_agent/` |
| **deliberative** | 神经符号推理与协调 | NeuroSymbolicReasoner, NeuroSymbolicWorldCoordinator, AutonomousThinkingEngine | `deliberative/` |
| **knowledge_rulebase** | 跨学科知识规则体系 | DisciplinaryRule, RuleRegistry, RuleIntegrationManager | `knowledge_rulebase/` |
| **core** | 架构治理核心层 | BaseModule, ConfigManager, ModuleRegistry | `core/` |
| **storage** | 存储与持久化 | StorageBackends, VersionedStorage, BackupManager | `storage/` |
| **monitoring** | 健康检查与指标 | HealthChecker, PrometheusMetrics | `monitoring/` |
| **homeostasis** | 内稳态系统 | HomeostasisManager, GoalGenerator | `homeostasis/` |
| **file_ingestion** | 文件导入与向量化 | FileIngestor, FeatureVectorizer, StructuredStorage | `file_ingestion/` |
| **webui** | Web用户界面 | FastAPI, WebSocket, HTML/CSS/JS | `webui/` |
| **agent** | 智能体核心 | SelfEvolvingAGI | `agent.py` |

### 核心算法

#### 1. 自由能原理 (Free Energy Principle)

智能体的主要优化目标是最小化变分自由能：

```
F = D_KL[q(z|x) || p(z)] - E_q[log p(x|z)]
```

通过预测值与观测值之间的均方误差(MSE)实现。

#### 2. 生长自编码器 (Growing Autoencoder)

自适应神经网络，动态调整结构：
- **生长**: 当自由能超过阈值时，添加神经元
- **剪枝**: 当自由能较低且网络过参数化时，移除神经元

#### 3. 脉冲神经网络 (Spiking Neural Network)

多型态SNN架构：
- **立体SNN** (Stereoscopic SNN) - 多层认知皮层模拟
- **增强SNN** (Enhanced SNN) - 跨层连接与储备池
- **生物听觉SNN** (Bio Auditory SNN) - 听觉通路模拟

#### 4. 神经符号推理 (Neuro-Symbolic Reasoning)

符号逻辑与神经网络深度融合：
- **符号编码**: 将符号映射到向量空间
- **关系推理**: 基于图的关系推理
- **规则引擎**: 可解释的符号规则推理
- **注意力机制**: 动态权重分配

#### 5. 世界模型 (World Model)

世界模型是智能体对环境的"内部心智模拟"，核心设计理念：
- **预测能力**: 根据当前状态和动作，推演环境未来的演化
- **泛化能力**: 在未见过的场景/物体/规则下，依然能做出合理判断
- **推理能力**: 支持反事实、因果式思考（"如果做A，会发生B吗"）

采用分层抽象 + 因果结构 + 多模态统一的核心思路，从感知到语义构建多级表征，兼顾预测精度与长程稳定性。

##### 5.1 多模态感知编码模块 (MultiModalEncoder)

深度融合视觉、文本、传感器、点云数据：
- **模态专用编码器**: 各模态独立编码路径，保留领域特异性特征
- **统一投影层**: 可学习的模态权重，实现跨模态信息对齐
- **模态注意力机制**: 动态分配不同模态的重要性权重

##### 5.2 分层抽象表征模块 (HierarchicalRepresentation)

四级表征层级，支持双向信息流动：
- **感知级** (Perceptual): 原始感官数据的初步编码
- **物体级** (Object): 实体特征提取与属性识别
- **场景级** (Scene): 实体间关系建模与空间布局
- **规则级** (Rule): 抽象规则提取与通用知识归纳

信息流向：自下而上特征聚合 + 自上而下约束修正

##### 5.3 世界动力学预测引擎 (DynamicsPredictor)

三层预测架构，核心模块：
- **微观物理预测器**: 细粒度状态变化预测（Gaussian Mixture Model概率输出）
- **中观物体预测器**: 物体级状态转移与交互预测
- **宏观语义预测器**: 高层语义演化与趋势预测
- **闭环修正机制**: 每N步进行预测误差校准与模型更新

##### 5.4 因果结构推理模块 (CausalReasoner)

因果发现与推理能力：
- **因果发现**: 从交互轨迹中自动识别实体间因果关系
- **干预式预测**: 基于do-calculus的干预效果预测
- **反事实推理**: "如果做A，会发生B吗"的假设性推理
- **分布外适应**: 对未见过场景的因果结构迁移

##### 5.5 记忆与知识库 (MemorySystem)

三级记忆系统：
- **工作记忆**: 近期状态缓存，支持实时推理
- **情景记忆**: 场景片段存储，向量检索匹配
- **语义记忆**: 通用规则蒸馏，从情景记忆中提取抽象知识

##### 5.6 规划与决策交互层 (PlanningInterface)

三大交互接口：
- **想象推理接口**: 生成想象轨迹，模拟多种可能的未来
- **目标导向接口**: 通过状态差异最小化实现目标导向规划
- **强化学习接口**: 生成RL轨迹，用于策略训练与优化

##### 5.7 训练范式

三阶段训练策略：
1. **自监督预训练**: 无标注数据上的预测学习，学习环境基本规律
2. **交互式微调**: 使用智能体交互轨迹进行微调，适应真实场景
3. **因果知识蒸馏**: 将因果发现结果蒸馏到预测模型中，提升可解释性

#### 6. 神经进化 (NEAT)

用于进化网络拓扑的遗传算法：
- 适应度函数: `1 / (free_energy + complexity_penalty)`
- 物种形成维持多样性
- 历史标记实现创新

#### 7. 元学习 (Multi-Armed Bandit)

自适应学习率选择：
- 探索-利用权衡
- 基于收敛速度的奖励
- 衰减探索率

---

## 📦 安装指南 / Installation

### 环境要求 / Requirements

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| PyTorch | 2.0+ | 深度学习框架 |
| NumPy | 1.24+ | 数值计算 |
| scipy | 1.10+ | 科学计算 |
| FastAPI | 0.100+ | Web框架 |
| Uvicorn | 0.23+ | ASGI服务器 |
| neat-python | 0.92+ | 神经进化算法 |
| psutil | 5.9+ | 系统资源监控 |
| matplotlib | 3.7+ | 可视化 |
| PyJWT | 2.8+ | JWT认证 |
| bcrypt | 4.0+ | 密码哈希 |
| httpx | 0.25+ | HTTP客户端 |
| faiss-cpu | 1.7+ | 向量检索 |

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/taojio/agi-agent.git
cd agi-agent

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 安装依赖
pip install -r agi_agent/requirements.txt
```

### GPU加速 (可选)

如需GPU加速，请确保已安装CUDA 11.0+：

```bash
# 安装GPU版本PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Docker部署

```bash
# 构建并启动
docker-compose up -d
```

---

## 🚀 使用教程 / Usage Tutorial

### 方式一：启动WebUI (推荐)

```bash
# 启动WebUI服务
python agi_agent/webui/app.py

# 或通过主入口
python main.py

# 访问WebUI
# 打开浏览器访问 http://localhost:8090
```

### 方式二：Python API调用

```python
from agi_agent.agent import SelfEvolvingAGI
import numpy as np

# 初始化智能体
agent = SelfEvolvingAGI(input_dim=16)

# 运行自主进化
report = agent.run(steps=1000)

# 打印结果
print(f"最终分数: {report['performance']['performance_score']['total_score']}")
print(f"知识规则数: {report['knowledge']['count']}")
```

### 方式三：命令行运行

```bash
# 运行智能体并生成报告
python agi_agent/run_agent.py --steps 1000 --output report.json
```

### WebUI界面功能

| 模块 | 功能描述 |
|------|----------|
| **多智能体聊天** | 与智能体进行实时对话 |
| **自我意识系统** | 查看自我意识指标和身份信息 |
| **记忆系统** | 查看各级记忆状态与统计 |
| **实时指标** | 自由能、置信度、新颖度等指标图表 |
| **自主思考** | 问题分解和批判性分析 |
| **决策系统** | 决策模拟器和统计 |
| **人格系统** | 人格特征和核心价值观 |
| **知识图谱** | 知识节点和学习记录 |
| **文件摄入** | 文件上传、搜索和管理 |
| **安全中心** | 安全概览与审计日志 |
| **自我改进** | 评估、诊断与改进提案 |
| **进化监控** | 进化状态与统计 |
| **实时日志** | 系统运行日志 |

### 登录与注册

系统内置JWT认证，首次使用需注册账号：
- 访问 `http://localhost:8090/register` 注册新用户
- 访问 `http://localhost:8090/login` 登录

### 文件摄入使用

通过WebUI或API上传文件：

```bash
# 使用curl上传文件
curl -X POST http://localhost:8090/api/file-ingestion/upload \
  -F "file=@document.pdf" \
  -F "file=@data.csv"
```

支持的文件格式：
- **文本**: TXT, MD, JSON, CSV, HTML
- **文档**: PDF, DOCX
- **音频**: MP3, WAV, FLAC
- **视频**: MP4, AVI, MOV

---

## 🔌 API 接口 / API Reference

### 认证API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/login` | GET | 登录页面 |
| `/register` | GET | 注册页面 |
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/refresh` | POST | 刷新令牌 |
| `/api/auth/logout` | POST | 用户登出 |
| `/api/auth/me` | GET | 获取当前用户信息 |

### 智能体API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/agent/status` | GET | 获取智能体状态 |
| `/api/agent/start` | POST | 启动智能体 |
| `/api/agent/stop` | POST | 停止智能体 |
| `/api/agent/step` | POST | 执行单步 |
| `/api/agent/run` | POST | 运行指定步数 |
| `/api/agent/report` | GET | 获取运行报告 |
| `/api/system/overview` | GET | 系统概览信息 |
| `/api/agents/list` | GET | 获取Agent列表 |
| `/api/sessions/list` | GET | 获取会话列表 |

### 记忆API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/memory/stats` | GET | 获取记忆统计 |
| `/api/memory/list` | GET | 获取记忆列表 |

### 安全API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/security/overview` | GET | 安全概览 |
| `/api/security/audit/list` | GET | 审计日志列表 |
| `/api/security/audit/stats` | GET | 审计统计 |

### 监控API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus指标 |

### 其他API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/chat/send` | POST | 发送聊天消息 |
| `/api/chat/history` | GET | 获取聊天历史 |
| `/api/file-ingestion/upload` | POST | 上传文件 |
| `/api/file-ingestion/search` | GET | 搜索文件内容 |
| `/api/self-awareness/status` | GET | 自我意识状态 |
| `/api/evolution/status` | GET | 进化状态 |
| `/api/decision/make` | POST | 执行决策 |
| `/api/plugins/available` | GET | 可用插件列表 |
| `/api/swarm/status` | GET | 集群状态 |

### WebSocket端点

| 端点 | 描述 |
|------|------|
| `/ws/metrics` | 实时指标推送 |
| `/ws/sensors` | 传感器数据推送 |
| `/ws/logs` | 实时日志推送 |

### API 示例

```bash
# 获取智能体状态
curl http://localhost:8090/api/agent/status

# 启动智能体
curl -X POST http://localhost:8090/api/agent/start

# 发送聊天消息
curl -X POST http://localhost:8090/api/chat/send \
  -H "Content-Type: application/json" \
  -d '{"content": "hello"}'

# 健康检查
curl http://localhost:8090/health

# Prometheus指标
curl http://localhost:8090/metrics
```

---

## 🛠️ 开发指南 / Development Guide

### 项目结构

```
agi_agent/
├── agent.py                     # 智能体核心类
├── run_agent.py                 # 运行入口
├── diagnostics.py               # 诊断工具
├── requirements.txt             # 依赖列表
├── ai_algorithms/               # AI算法库
│   ├── anomaly_detection.py     # 异常检测
│   ├── clustering.py            # 聚类算法
│   ├── feature_engineering.py   # 特征工程
│   ├── forecasting.py           # 预测算法
│   ├── pattern_recognition.py   # 模式识别
│   └── scheduling.py            # 调度算法
├── analysis/                    # 分析引擎
│   ├── dimensions.py            # 维度分析
│   ├── engine.py                # 分析引擎
│   └── metrics.py               # 指标计算
├── automation/                  # 自动化引擎
│   └── engine.py                # 自动化引擎
├── autonomous_action/           # 自主行动
│   ├── action_executor.py       # 行动执行器
│   ├── action_orchestrator.py   # 行动编排器
│   ├── active_explorer.py       # 主动探索器
│   ├── path_planner.py          # 路径规划器
│   └── target_decomposer.py     # 目标分解器
├── chat/                        # 聊天系统
│   ├── chat_server.py           # 聊天服务器
│   ├── message_store.py         # 消息存储
│   └── permission_manager.py    # 权限管理器
├── cognitive/                   # 认知系统
│   ├── dual_system.py           # 双系统认知
│   ├── inference_engine.py      # 推理引擎
│   ├── causal_reasoning.py      # 因果推理
│   ├── orchestrator.py          # 认知编排器
│   ├── self_model.py            # 自我模型
│   ├── stereoscopic_snn.py      # 立体SNN
│   ├── enhanced_snn.py          # 增强SNN
│   ├── bio_auditory_snn.py      # 生物听觉SNN
│   ├── module_synaptic_bus.py   # 模块突触总线
│   ├── world_model.py           # 世界模型引擎
│   ├── predictive_coding.py     # 预测编码
│   └── context_awareness.py     # 上下文感知
├── config/                      # 配置模块
│   └── settings.py              # 全局配置参数
├── core/                        # 架构治理核心层
│   ├── base_module.py           # 模块基类(生命周期管理)
│   ├── config.py                # 统一配置管理
│   ├── exceptions.py            # 统一异常体系
│   └── registry.py              # 模块注册中心
├── cultivation/                 # 培养系统
│   └── cultivation_manager.py   # 培养管理器
├── data_standards/              # 数据标准
│   ├── models.py                # 数据模型
│   ├── serialization.py         # 序列化
│   └── validator.py             # 验证器
├── decision/                    # 决策系统
│   ├── decision_engine.py       # 决策引擎
│   ├── action_planner.py        # 行动规划
│   ├── decision_strategies.py   # 决策策略
│   ├── decision_tracker.py      # 决策追踪
│   ├── execution_monitor.py     # 执行监控
│   ├── risk_assessment.py       # 风险评估
│   └── world_model_bridge.py    # 世界模型桥接
├── deliberative/                # 神经符号推理系统
│   ├── neuro_symbolic_reasoner.py           # 神经符号推理器
│   ├── neuro_symbolic_world_coordinator.py  # 世界模型协调器
│   ├── autonomous_thinking_engine.py        # 自主思考引擎
│   ├── advanced_reasoner.py                 # 高级推理器
│   ├── causal_reasoner.py                   # 因果推理器
│   ├── decision_optimizer.py                # 决策优化器
│   ├── hypothesis_generator.py              # 假设生成器
│   ├── logical_deductor.py                  # 逻辑演绎器
│   ├── problem_formulator.py               # 问题公式化
│   ├── simulation_engine.py                # 模拟引擎
│   └── thinking_orchestrator.py            # 思考编排器
├── embodied/                    # 实体化
│   ├── installer.py             # 安装器
│   ├── resource_manager.py      # 资源管理器
│   └── system_tray.py           # 系统托盘
├── evaluation/                  # 评估系统
│   ├── evaluator.py             # 评估器
│   └── visualizer.py            # 可视化器
├── evolution/                   # 进化引擎
│   ├── quad_level_evolution.py  # 四级进化
│   ├── dual_loop_evolution.py   # 双循环进化
│   ├── neat_engine.py           # NEAT进化
│   └── metaskill_generator.py   # 元技能生成器
├── file_ingestion/              # 文件摄入系统
│   ├── file_ingestor.py         # 摄入管理器
│   ├── file_parsers.py          # 文件解析器
│   ├── preprocessor.py          # 数据预处理
│   ├── vectorization.py         # 特征向量化
│   ├── structured_storage.py    # 结构化存储
│   └── file_access.py           # 文件访问管理
├── homeostasis/                 # 内稳态系统
│   └── homeostasis.py           # 内稳态管理器
├── knowledge_rulebase/          # 跨学科知识规则体系
│   ├── disciplinary_rule.py     # 规则数据模型
│   ├── rule_registry.py         # 规则注册中心
│   ├── physics_rules.py         # 物理规则集
│   ├── math_rules.py            # 数学规则集
│   ├── chemistry_rules.py       # 化学规则集
│   ├── biology_rules.py         # 生物规则集
│   ├── chinese_rules.py         # 语文规则集
│   └── integration.py           # 跨学科融合
├── learning/                    # 学习模块
│   ├── knowledge_graph.py       # 知识图谱
│   ├── knowledge_ingestor.py    # 知识摄入
│   ├── knowledge_integrator.py  # 知识整合
│   ├── experience_distiller.py  # 经验蒸馏
│   ├── learning_planner.py      # 学习规划
│   ├── online_learner.py        # 在线学习
│   └── meta_learning.py         # 元学习
├── meta_cognitive/              # 元认知系统
│   ├── meta_cognitive_orchestrator.py  # 元认知编排器
│   ├── self_model.py            # 自我模型
│   ├── reflection_engine.py     # 反思引擎
│   ├── cognitive_monitor.py     # 认知监控
│   ├── cognitive_regulator.py   # 认知调节
│   ├── capability_assessor.py   # 能力评估
│   ├── boundary_guardian.py     # 边界守护
│   ├── strategy_regulator.py    # 策略调节
│   ├── learning_self_regulation.py  # 学习自我调节
│   ├── legacy_layer.py          # 遗留层
│   └── meta_learning_engine.py  # 元学习引擎
├── meta_decision/               # 元决策系统
│   ├── orchestrator.py          # 编排器
│   ├── decision_monitor.py      # 决策监控
│   ├── decision_optimizer.py    # 决策优化
│   └── quality_analyzer.py      # 质量分析
├── meta_evolution/              # 元进化系统
│   ├── orchestrator.py          # 编排器
│   ├── evolution_controller.py  # 进化控制器
│   ├── gene_pool.py             # 基因池
│   ├── genetic_algorithm.py     # 遗传算法
│   ├── parameter_optimizer.py   # 参数优化
│   └── structural_optimizer.py # 结构优化
├── meta_learning/               # 元学习系统
│   ├── orchestrator.py          # 编排器
│   ├── meta_learner.py          # 元学习器
│   ├── meta_knowledge_base.py   # 元知识库
│   ├── strategy_optimizer.py    # 策略优化
│   └── task_adaptation.py       # 任务适应
├── meta_parsing/                # 元解析系统
│   ├── orchestrator.py          # 编排器
│   ├── meta_parser.py           # 元解析器
│   ├── complex_data_processor.py # 复杂数据处理
│   └── data_transformer.py      # 数据转换器
├── meta_programming/            # 元编程系统
│   ├── orchestrator.py          # 编排器
│   ├── code_analyzer.py         # 代码分析器
│   ├── code_generator.py        # 代码生成器
│   └── dynamic_executor.py      # 动态执行器
├── module_bus/                  # 模块总线
│   ├── bus.py                   # 总线核心
│   ├── data_stream.py           # 数据流
│   ├── event.py                 # 事件系统
│   ├── message.py               # 消息定义
│   └── service_registry.py      # 服务注册
├── monitoring/                  # 监控运维
│   └── health.py                # 健康检查与指标
├── multi_agent/                 # 多智能体协作
│   ├── agent_swarm.py           # 智能体集群
│   ├── task_allocation.py       # 任务分配
│   ├── shared_memory.py         # 共享记忆空间
│   ├── conflict_resolver.py     # 冲突解决
│   ├── hierarchical_dispatcher.py # 层级调度器
│   └── workspace.py             # 工作空间
├── orchestration/               # 统一编排层 v2
│   ├── event_bus.py             # 事件总线
│   ├── task_dag.py              # 任务DAG图
│   ├── orchestrator.py          # 编排引擎
│   └── automation_linkage.py    # 自动化联动
├── personality/                 # 人格系统
│   └── personality_core.py      # 人格核心
├── perception/                  # 感知系统
│   ├── autoencoder.py           # 生长自编码器
│   └── multimodal_fusion.py     # 多模态融合
├── plugins/                     # 插件系统
│   ├── plugin_base.py           # 插件基类
│   ├── plugin_manager.py        # 插件管理器
│   └── mods/                    # 内置插件
│       ├── data_processor.py    # 数据处理器
│       ├── noise_sensor.py      # 噪声传感器
│       ├── temperature_sensor.py # 温度传感器
│       ├── vector_database.py   # 向量数据库
│       ├── web_crawler.py       # 网页爬虫
│       └── worm_evolve_sim.py   # 蠕虫进化模拟
├── reflex/                      # 反射系统
│   ├── reflex_controller.py     # 反射控制器
│   ├── instinct_actions.py      # 本能动作
│   ├── pattern_matcher.py       # 模式匹配器
│   ├── rule_engine.py           # 规则引擎
│   └── spiking_core.py          # 脉冲核心
├── security/                    # 安全认证体系
│   ├── jwt_auth.py              # JWT认证
│   ├── rbac.py                  # RBAC权限管理
│   ├── rate_limiter.py          # 速率限制
│   ├── validation.py            # 输入验证
│   ├── headers.py               # 安全响应头
│   ├── audit_logger.py          # 审计日志
│   ├── audit_trail.py           # 审计追踪
│   ├── safety_monitor.py        # 安全监控
│   ├── compliance_checker.py    # 合规检查
│   ├── hard_boundary.py         # 硬边界
│   ├── circuit_breaker.py       # 熔断器
│   └── risk_classifier.py       # 风险分类
├── self_improvement/            # 自我改进系统
│   ├── self_improver.py         # 递归自我改进
│   ├── bootstrapped_improver.py # 引导式自改进
│   ├── automated_improvement_loop.py # 自动化改进循环
│   ├── feedback_loop.py         # 反馈循环
│   ├── performance_baseline.py  # 性能基线
│   ├── performance_evaluator.py # 性能评估器
│   ├── regression_validator.py  # 回归验证器
│   ├── safety_verifier.py       # 安全验证器
│   ├── self_diagnostic.py       # 自我诊断
│   ├── symbolic_self_model.py   # 符号自我模型
│   ├── symbolic_verifier.py     # 符号验证器
│   └── tiered_modification.py   # 层级修改
├── skills/                      # 技能管理
│   ├── skills_manager.py        # 技能管理器
│   ├── calendar/                # 日历技能
│   ├── fullstack-companion/     # 全栈开发技能
│   ├── recursive-self-improvement/  # 递归自我改进技能
│   ├── self-improving/          # 自我改进技能
│   └── self-improving-1.2.16/   # 自我改进技能(版本)
├── soul/                        # 灵魂系统
│   ├── soul_model.py            # 灵魂模型
│   └── soul_parser.py           # 灵魂解析器
├── storage/                     # 存储与持久化
│   ├── backends.py              # 存储后端
│   ├── backup.py                # 备份管理
│   ├── persistence.py           # 持久化
│   └── state_manager.py         # 状态管理
├── task_engine/                 # 任务引擎
│   ├── async_task_board.py      # 异步任务板
│   ├── checkpoint_manager.py    # 检查点管理
│   ├── dag_engine.py            # DAG引擎
│   └── heartbeat_scheduler.py   # 心跳调度器
├── tests/                       # 测试用例
│   ├── test_enhanced_capabilities.py    # 增强能力测试
│   └── test_neuro_symbolic_world.py    # 神经符号与世界模型测试
├── utils/                       # 工具类
│   ├── logger.py                # 日志工具
│   ├── metrics.py               # 指标工具
│   └── numpy_utils.py           # NumPy工具
├── webui/                       # WebUI界面
│   ├── app.py                   # FastAPI主应用(端口8090)
│   ├── api_server.py            # API服务器(端口8000)
│   ├── index.html               # 主页面
│   ├── settings.json            # WebUI设置
│   └── static/                  # 静态资源
│       ├── index.html           # 主页面
│       ├── login.html           # 登录页面
│       ├── register.html        # 注册页面
│       ├── app.js               # 前端逻辑
│       ├── style.css            # 样式表
│       ├── i18n.js              # 国际化
│       └── locales/             # 语言资源
│           ├── zh.json          # 中文翻译
│           └── en.json          # 英文翻译
└── docs/                        # 文档
    ├── technical_documentation.md  # 技术文档
    └── user_manual.md           # 用户手册
```

### 配置说明

配置文件位于 `agi_agent/config/settings.py` 和 `agi_agent/webui/settings.json`：

```python
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 核心阈值
FREE_ENERGY_THRESHOLD = 0.3    # 自由能阈值
EVOLVE_TRIGGER_STEP = 200      # 进化触发步数
NOVELTY_THRESHOLD = 0.5        # 新颖度阈值

# 记忆设置
MEMORY_BUFFER_SIZE = 200       # 记忆缓冲区大小
KNOWLEDGE_MAX_SIZE = 1000      # 知识最大容量

# 学习率池
LEARNING_RATE_POOL = [1e-4, 5e-4, 1e-3, 2e-3]

# 安全约束
SAFETY_MAX_ENERGY = 10.0       # 最大自由能
SAFETY_MAX_MEMORY_GB = 4.0     # 最大内存使用
SAFETY_MAX_GPU_UTIL = 0.95     # 最大GPU利用率
SAFETY_MAX_LATENCY_MS = 1000   # 最大延迟
```

WebUI设置 (`settings.json`)：
```json
{
  "input_dim": 512,
  "auto_start": true,
  "save_interval": 1000,
  "sensor_enabled": true,
  "theme": "dark"
}
```

### 插件开发

参考 [PLUGIN_DEVELOPMENT_SPEC.md](agi_agent/plugins/PLUGIN_DEVELOPMENT_SPEC.md) 了解插件开发规范。

### 记忆插件开发

```python
from agi_agent.memory import MemoryPlugin, PluginType

class CustomRetrievalPlugin(MemoryPlugin):
    def get_name(self) -> str:
        return "custom_retrieval"
    
    def get_type(self) -> PluginType:
        return PluginType.RETRIEVAL
    
    def process(self, entry) -> MemoryEntry:
        return entry
```

### 技能开发

技能位于 `agi_agent/skills/` 目录，每个技能包含：
- `SKILL.md` - 技能定义和元数据
- 相关文档文件

### 国际化

项目支持中英双语，语言资源文件位于 `agi_agent/webui/static/locales/`：
- `zh.json` - 中文翻译
- `en.json` - 英文翻译

---

## ✅ 测试验证 / Testing

### 运行测试

```bash
# 运行所有测试
python -m pytest agi_agent/tests/ -v

# 运行增强能力测试
python -m agi_agent.tests.test_enhanced_capabilities

# 运行神经符号与世界模型测试
python -m agi_agent.tests.test_neuro_symbolic_world
```

### 测试覆盖

| 测试模块 | 覆盖范围 |
|----------|----------|
| **test_enhanced_capabilities.py** | 增强能力测试：抽象引擎、高级推理、情景感知、知识图谱、主动学习 |
| **test_neuro_symbolic_world.py** | 神经符号推理与世界模型测试：符号推理、因果发现、情景模拟、反事实推理、协调机制 |

---

## 🛡️ 安全框架 / Security Framework

### JWT认证

- **双令牌机制**: access token (15分钟) + refresh token (7天)
- **密码安全**: bcrypt哈希存储
- **令牌黑名单**: 登出时令牌加入黑名单

### RBAC权限模型

| 角色 | 权限级别 | 说明 |
|------|----------|------|
| SUPER_ADMIN | 最高 | 全部权限 |
| ADMIN | 高 | 系统管理 |
| OPERATOR | 中 | 日常操作 |
| VIEWER | 低 | 只读访问 |
| GUEST | 最低 | 受限访问 |

### 安全约束

| 约束项 | 阈值 | 严重程度 | 触发动作 |
|--------|------|----------|----------|
| 自由能 | > 10.0 | Critical | 紧急停机 |
| 内存使用 | > 4GB | Warning | 限流 |
| GPU利用率 | > 95% | Warning | 限流 |
| 延迟 | > 1000ms | Info | 记录日志 |

### 合规检查

- **偏差检测**: 特征-行动相关性监控
- **数据隐私**: 敏感模式检测
- **透明度**: 决策轨迹日志
- **问责制**: 审计追踪维护

### 安全模块

| 模块 | 功能 |
|------|------|
| **SafetyMonitor** | 实时安全约束监控 |
| **ComplianceChecker** | 合规性检查 |
| **HardBoundarySystem** | 硬边界保护 |
| **RiskClassifier** | 风险等级分类 |
| **CircuitBreaker** | 熔断机制 |
| **AuditTrail** | 审计追踪 |
| **JWTAuth** | JWT认证 |
| **RBAC** | 权限管理 |
| **RateLimiter** | 速率限制 |
| **Validation** | 输入验证 |

---

## 升级项目 / Project Phoenix

系统正在进行全栈升级（Project Phoenix），详见 [upgrade_project/](upgrade_project/) 目录：

| 模块 | 状态 | 规格文档 |
|------|------|----------|
| 安全系统升级 | 已完成 | [UPG-001](upgrade_project/specs/UPG-001_security_upgrade_spec.md) |
| 测试体系建设 | 已完成 | [UPG-002](upgrade_project/specs/UPG-002_testing_system_spec.md) |
| 架构治理核心层 | 已完成 | [UPG-003](upgrade_project/specs/UPG-003_architecture_governance_spec.md) |
| 记忆底座 v2.0 | 已完成 | [UPG-004](upgrade_project/specs/UPG-004_memory_v2_spec.md) |
| 存储与持久化 | 已完成 | [UPG-005](upgrade_project/specs/UPG-005_storage_upgrade_spec.md) |
| 统一编排层 v2 | 已完成 | [UPG-006](upgrade_project/specs/UPG-006_orchestration_spec.md) |
| WebUI现代化 | 已完成 | [UPG-007](upgrade_project/specs/UPG-007_webui_spec.md) |
| 部署与运维 | 已完成 | [UPG-008](upgrade_project/specs/UPG-008_deployment_spec.md) |
| 认知推理引擎 | 已完成 | [UPG-009](upgrade_project/specs/UPG-009_cognitive_spec.md) |
| 多智能体协作 | 已完成 | [UPG-010](upgrade_project/specs/UPG-010_multi_agent_spec.md) |
| 神经符号推理与世界模型 | 已完成 | 新增模块 |
| 跨学科知识规则体系 | 已完成 | `knowledge_rulebase/` |
| 进化系统 v3 | 规划中 | [UPG-011](upgrade_project/specs/UPG-011_evolution_spec.md) |
| 自我改进系统 | 规划中 | [UPG-012](upgrade_project/specs/UPG-012_self_improvement_spec.md) |
| 多模态感知 | 规划中 | [UPG-013](upgrade_project/specs/UPG-013_multimodal_spec.md) |
| 插件生态平台 | 规划中 | [UPG-014](upgrade_project/specs/UPG-014_plugin_spec.md) |

---

## 🤝 贡献指南 / Contributing

欢迎贡献代码！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 获取详细信息。

### 贡献流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 使用类型提示
- 添加适当的文档字符串
- 编写单元测试

---

## 📄 许可证 / License

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

## 📞 联系方式 / Contact

- 项目地址: https://github.com/taojio/agi-agent
- 问题反馈: https://github.com/taojio/agi-agent/issues

---

**Made with ❤️ by the AGI Agent Team**