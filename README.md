# AGI Agent - Self-Evolving Autonomous Intelligence

🤖 一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统

## 项目简介 / Project Description

AGI Agent 是一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统。它采用先进的认知架构，能够持续学习、自适应进化，并在复杂环境中做出自主决策。系统内置五级记忆底座、脉冲神经网络（SNN）认知引擎、多智能体协作网络、JWT安全认证体系、神经符号推理系统、世界模型，以及完整的自我改进与进化机制。

---

## 📋 目录 / Table of Contents

- [项目简介 / Project Description](#项目简介--project-description)
- [核心特性 / Core Features](#核心特性--core-features)
- [技术架构 / Technical Architecture](#技术架构--technical-architecture)
- [安装指南 / Installation](#安装指南--installation)
- [使用教程 / Usage Tutorial](#使用教程--usage-tutorial)
- [API 接口 / API Reference](#api-接口--api-reference)
- [开发指南 / Development Guide](#开发指南--development-guide)
- [测试验证 / Testing](#测试验证--testing)
- [安全框架 / Security Framework](#安全框架--security-framework)
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
│   │  AutoEncoder│    │  SNN/Dual   │    │  Action     │             │
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
│   │  知识图谱   │    │  元认知系统  │                                │
│   │  Knowledge  │    │  Meta-Cog   │                                │
│   │  Graph      │    │  Orchestrator│                               │
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
└──────────────────────────────────────────────────────────────────────┘
```

### 模块详解

| 模块 | 职责 | 核心组件 | 文件位置 |
|------|------|----------|----------|
| **perception** | 感官输入处理与特征提取 | GrowingAutoEncoder, MultimodalFusion | `perception/` |
| **cognitive** | 时序预测与自主推理 | DualSystemCognition, CausalReasoningEngine, SNN, WorldModelEngine, MultiModalEncoder, HierarchicalRepresentation, DynamicsPredictor, CausalReasoner, MemorySystem, PlanningInterface | `cognitive/` |
| **memory** | 工业级记忆底座 v2.0 | MemoryHarness, MemoryStore, PluginRegistry, StorageBackend | `memory/` |
| **learning** | 元学习与知识管理 | MetaLearningLayer, KnowledgeGraph | `learning/` |
| **decision** | 自主决策与行动规划 | AutonomousDecisionEngine, ActionPlanner | `decision/` |
| **evolution** | 神经进化与结构优化 | EvolutionEngine (NEAT), QuadLevelEvolution | `evolution/` |
| **meta_cognitive** | 自我监控与资源调度 | MetaCognitiveOrchestrator, SelfModel | `meta_cognitive/` |
| **self_improvement** | 递归自我改进 | RecursiveSelfImprover, BootstrappedSelfImprover | `self_improvement/` |
| **security** | 安全认证与合规检查 | JWTAuth, RBAC, RateLimiter, AuditLogger, SafetyMonitor | `security/` |
| **orchestration** | 统一编排层 v2 | EventBus, TaskDAG, Orchestrator | `orchestration/` |
| **multi_agent** | 多智能体协作 | AgentSwarm, TaskAllocator, SharedMemorySpace | `multi_agent/` |
| **deliberative** | 神经符号推理与协调 | NeuroSymbolicReasoner, NeuroSymbolicWorldCoordinator | `deliberative/` |
| **core** | 架构治理核心层 | BaseModule, ConfigManager, ModuleRegistry | `core/` |
| **storage** | 存储与持久化 | StorageBackends, VersionedStorage, BackupManager | `storage/` |
| **monitoring** | 健康检查与指标 | HealthChecker, PrometheusMetrics | `monitoring/` |
| **homeostasis** | 内稳态系统 | HomeostasisManager, GoalGenerator | `homeostasis/` |
| **file_ingestion** | 文件导入与向量化 | FileIngestor, FeatureVectorizer, StructuredStorage | `file_ingestion/` |
| **webui** | Web用户界面 | FastAPI, WebSocket, HTML/CSS/JS | `webui/` |

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
├── agent.py                  # 智能体核心类
├── run_agent.py              # 运行入口
├── diagnostics.py            # 诊断工具
├── requirements.txt          # 依赖列表
├── config/                   # 配置模块
│   └── settings.py           # 全局配置参数
├── core/                     # 架构治理核心层
│   ├── base_module.py        # 模块基类(生命周期管理)
│   ├── config.py             # 统一配置管理
│   ├── exceptions.py         # 统一异常体系
│   └── registry.py           # 模块注册中心
├── perception/               # 感知系统
│   ├── autoencoder.py        # 生长自编码器
│   └── multimodal_fusion.py  # 多模态融合
├── cognitive/                # 认知系统
│   ├── dual_system.py        # 双系统认知
│   ├── inference_engine.py   # 推理引擎
│   ├── causal_reasoning.py   # 因果推理
│   ├── orchestrator.py       # 认知编排器
│   ├── self_model.py         # 自我模型
│   ├── stereoscopic_snn.py   # 立体SNN
│   ├── enhanced_snn.py       # 增强SNN
│   ├── bio_auditory_snn.py   # 生物听觉SNN
│   ├── module_synaptic_bus.py # 模块突触总线
│   └── world_model.py        # 世界模型引擎（包含MultiModalEncoder、HierarchicalRepresentation、DynamicsPredictor、CausalReasoner、MemorySystem、PlanningInterface）
├── deliberative/             # 神经符号推理系统
│   ├── neuro_symbolic_reasoner.py    # 神经符号推理器
│   └── neuro_symbolic_world_coordinator.py  # 世界模型协调器
├── memory/                   # 记忆底座 v2.0
│   ├── memory_tiers.py       # 记忆层级与数据模型
│   ├── memory_store.py       # 分层存储实现
│   ├── memory_harness.py     # 检索与调度管理器
│   ├── retrieval.py          # 语义检索插件
│   ├── forgetting.py         # 自适应遗忘插件
│   ├── consolidation.py      # 记忆巩固插件
│   ├── plugins.py            # 插件系统(MemoryPlugin+Registry)
│   ├── payload.py            # 多模态数据载荷
│   ├── storage_backend.py    # 存储后端抽象层
│   └── security.py           # 记忆安全层(权限+脱敏)
├── learning/                 # 学习模块
│   ├── knowledge_graph.py    # 知识图谱
│   └── meta_learning.py      # 元学习
├── decision/                 # 决策系统
│   ├── decision_engine.py    # 决策引擎
│   └── action_planner.py     # 行动规划
├── evolution/                # 进化引擎
│   ├── quad_level_evolution.py  # 四级进化
│   ├── dual_loop_evolution.py   # 双循环进化
│   └── neat_engine.py       # NEAT进化
├── meta_cognitive/           # 元认知系统
│   ├── meta_cognitive_orchestrator.py
│   └── self_model.py         # 自我模型
├── self_improvement/         # 自我改进系统
│   ├── self_improver.py      # 递归自我改进
│   ├── bootstrapped_improver.py # 引导式自改进
│   └── safety_verifier.py    # 安全验证器
├── security/                 # 安全认证体系
│   ├── jwt_auth.py           # JWT认证
│   ├── rbac.py               # RBAC权限管理
│   ├── rate_limiter.py       # 速率限制
│   ├── validation.py         # 输入验证
│   ├── headers.py            # 安全响应头
│   ├── audit_logger.py       # 审计日志
│   ├── safety_monitor.py     # 安全监控
│   ├── compliance_checker.py # 合规检查
│   ├── hard_boundary.py      # 硬边界
│   └── circuit_breaker.py    # 熔断器
├── orchestration/            # 统一编排层 v2
│   ├── event_bus.py          # 事件总线
│   ├── task_dag.py           # 任务DAG图
│   └── orchestrator.py       # 编排引擎
├── multi_agent/              # 多智能体协作
│   ├── agent_swarm.py        # 智能体集群
│   ├── task_allocation.py    # 任务分配
│   ├── shared_memory.py      # 共享记忆空间
│   └── conflict_resolver.py  # 冲突解决
├── storage/                  # 存储与持久化
│   ├── backends.py           # 存储后端
│   ├── backup.py             # 备份管理
│   └── state_manager.py      # 状态管理
├── monitoring/               # 监控运维
│   └── health.py             # 健康检查与指标
├── homeostasis/              # 内稳态系统
│   └── homeostasis.py        # 内稳态管理器
├── file_ingestion/           # 文件摄入系统
│   ├── file_ingestor.py      # 摄入管理器
│   ├── file_parsers.py       # 文件解析器
│   └── vectorization.py      # 特征向量化
├── plugins/                  # 插件系统
│   ├── plugin_base.py        # 插件基类
│   ├── plugin_manager.py     # 插件管理器
│   └── mods/                 # 内置插件
├── skills/                   # 技能管理
│   └── skills_manager.py     # 技能管理器
├── webui/                    # WebUI界面
│   ├── app.py                # FastAPI主应用(端口8090)
│   ├── api_server.py         # API服务器(端口8000)
│   └── static/               # 静态资源
│       ├── index.html        # 主页面
│       ├── login.html        # 登录页面
│       ├── register.html     # 注册页面
│       ├── app.js            # 前端逻辑
│       └── style.css         # 样式表
└── tests/                    # 测试用例
    ├── conftest.py           # pytest配置与fixtures
    ├── unit/                 # 单元测试
    │   ├── test_security/    # 安全模块测试
    │   ├── test_core/        # 核心层测试
    │   ├── test_memory/      # 记忆模块测试
    │   ├── test_orchestration/ # 编排层测试
    │   └── test_storage/     # 存储模块测试
    ├── integration/          # 集成测试
    │   ├── test_auth_api.py  # 认证API测试
    │   └── test_security_api.py # 安全API测试
    ├── test_enhanced_capabilities.py  # 增强能力测试
    └── test_neuro_symbolic_world.py   # 神经符号与世界模型测试
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
        # 自定义检索逻辑
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

# 运行单元测试
python -m pytest agi_agent/tests/unit/ -v

# 运行集成测试
python -m pytest agi_agent/tests/integration/ -v

# 运行特定模块测试
python -m pytest agi_agent/tests/unit/test_security/ -v
python -m pytest agi_agent/tests/unit/test_memory/ -v
python -m pytest agi_agent/tests/unit/test_orchestration/ -v

# 运行增强能力测试
python -m agi_agent.tests.test_enhanced_capabilities

# 运行神经符号与世界模型测试
python -m agi_agent.tests.test_neuro_symbolic_world
```

### 测试覆盖

| 测试模块 | 覆盖范围 |
|----------|----------|
| **test_security/** | JWT认证、RBAC权限、速率限制、输入验证、审计日志 |
| **test_core/** | 模块基类、配置管理、注册中心 |
| **test_memory/** | 语义检索、自适应遗忘、记忆巩固 |
| **test_orchestration/** | 事件总线、任务DAG、编排引擎 |
| **test_storage/** | 存储后端、备份管理 |
| **test_auth_api** | 注册、登录、令牌刷新等API集成测试 |
| **test_security_api** | 安全概览、审计日志等API集成测试 |
| **test_functional.py** | 功能测试：感知、认知、学习、进化、执行 |
| **test_performance.py** | 性能测试：延迟、吞吐量、内存稳定性 |
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
