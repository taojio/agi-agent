# AGI Agent - Self-Evolving Autonomous Intelligence

> 🤖 一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统

## 项目简介 / Project Description

AGI Agent 是一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统。它采用先进的认知架构，能够持续学习、自适应进化，并在复杂环境中做出自主决策。

AGI Agent is an artificial intelligence agent system with self-awareness, autonomous thinking, and independent action capabilities. It adopts an advanced cognitive architecture that enables continuous learning, adaptive evolution, and autonomous decision-making in complex environments.

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

### 记忆与学习 / Memory & Learning
- **五级记忆系统** (L1-L5 Memory) - 从感官记忆到永久记忆的完整层级
- **知识图谱** (Knowledge Graph) - 结构化知识存储与推理
- **元学习** (Meta-Learning) - 自适应学习策略与超参数优化
- **文件摄入系统** (File Ingestion) - 支持多类型文件的导入、处理与向量化

### 决策与行动 / Decision & Action
- **自主决策引擎** (Autonomous Decision Engine) - 基于风险评估的智能决策
- **目标分解** (Target Decomposition) - 复杂任务的层次化分解
- **路径规划** (Path Planning) - 行动路径的优化规划
- **主动探索** (Active Exploration) - 未知领域的自主探索

### 进化与自我改进 / Evolution & Self-Improvement
- **四级进化机制** (Quad-Level Evolution) - 微观到宏观的多层次进化
- **递归自我改进** (Recursive Self-Improvement) - 持续自我优化能力
- **架构变异** (Architecture Mutation) - 神经网络结构的自适应调整
- **性能评估** (Performance Evaluation) - 实时性能监控与改进

### 安全与合规 / Safety & Compliance
- **多层次安全防护** (Multi-layer Safety) - 硬边界、合规检查、风险分类
- **安全监控** (Safety Monitor) - 实时安全约束与紧急停机
- **审计追踪** (Audit Trail) - 完整的决策过程记录

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
│   │  记忆系统   │    │  决策引擎   │    │  人格系统   │             │
│   │  Memory     │    │  Decision   │    │  Personality│             │
│   │  L1-L5      │    │  Engine     │    │  Core       │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                                          │
│        ▼                  ▼                                          │
│   ┌─────────────┐    ┌─────────────┐                                │
│   │  知识图谱   │    │  元认知系统  │                                │
│   │  Knowledge  │    │  Meta-Cog   │                                │
│   │  Graph      │    │  Orchestrator│                               │
│   └─────────────┘    └─────────────┘                                │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  进化引擎   │    │  自我改进   │    │  安全监控   │             │
│   │  Evolution  │    │  Self-Improve│   │  Security   │             │
│   │  Engine     │    │  Engine     │    │  Monitor    │             │
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
| **cognitive** | 时序预测与自主推理 | DualSystemCognition, CausalReasoningEngine | `cognitive/` |
| **memory** | 五级记忆管理 | MemoryHarness, MemoryStore | `memory/` |
| **learning** | 元学习与知识管理 | MetaLearningLayer, KnowledgeGraph | `learning/` |
| **decision** | 自主决策与行动规划 | AutonomousDecisionEngine, ActionPlanner | `decision/` |
| **evolution** | 神经进化与结构优化 | EvolutionEngine (NEAT), QuadLevelEvolution | `evolution/` |
| **meta_cognitive** | 自我监控与资源调度 | MetaCognitiveOrchestrator, SelfModel | `meta_cognitive/` |
| **self_improvement** | 递归自我改进 | RecursiveSelfImprover, BootstrappedSelfImprover | `self_improvement/` |
| **security** | 安全约束与合规检查 | SafetyMonitor, ComplianceChecker, HardBoundarySystem | `security/` |
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

#### 3. 分层预测编码 (Hierarchical Predictive Coding)

多层时序预测网络：
- 前馈预测生成
- 反向误差传播
- 多步前瞻推理

#### 4. 神经进化 (NEAT)

用于进化网络拓扑的遗传算法：
- 适应度函数: `1 / (free_energy + complexity_penalty)`
- 物种形成维持多样性
- 历史标记实现创新

#### 5. 元学习 (Multi-Armed Bandit)

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
pip install -r requirements.txt
```

### GPU加速 (可选)

如需GPU加速，请确保已安装CUDA 11.0+：

```bash
# 安装GPU版本PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 使用教程 / Usage Tutorial

### 方式一：启动WebUI (推荐)

```bash
# 启动WebUI服务
python agi_agent/webui/app.py

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

WebUI提供以下功能模块：

| 模块 | 功能描述 |
|------|----------|
| **💬 多智能体聊天** | 与智能体进行实时对话 |
| **🧠 自我意识系统** | 查看自我意识指标和身份信息 |
| **🗄️ 记忆系统** | 查看各级记忆状态 |
| **📊 实时指标** | 自由能、置信度、新颖度等指标图表 |
| **🤔 自主思考** | 问题分解和批判性分析 |
| **⚖️ 决策系统** | 决策模拟器和统计 |
| **👤 人格系统** | 人格特征和核心价值观 |
| **🌐 知识图谱** | 知识节点和学习记录 |
| **📚 文件摄入** | 文件上传、搜索和管理 |
| **📝 实时日志** | 系统运行日志 |

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

### REST API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/agent/status` | GET | 获取智能体状态 |
| `/api/agent/start` | POST | 启动智能体 |
| `/api/agent/stop` | POST | 停止智能体 |
| `/api/agent/introspect` | POST | 触发自我反思 |
| `/api/file-ingestion/upload` | POST | 上传文件 |
| `/api/file-ingestion/search` | GET | 搜索文件内容 |
| `/api/file-ingestion/records` | GET | 获取摄入记录列表 |
| `/api/file-ingestion/stats` | GET | 获取统计信息 |
| `/api/self-awareness/metrics` | GET | 获取自我意识指标 |
| `/api/thinking/status` | GET | 获取思考状态 |
| `/api/decision/make` | POST | 执行决策 |
| `/api/knowledge/graph` | GET | 获取知识图谱 |

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

# 搜索文件
curl "http://localhost:8090/api/file-ingestion/search?q=人工智能"

# 获取自我意识指标
curl http://localhost:8090/api/self-awareness/metrics
```

---

## 🛠️ 开发指南 / Development Guide

### 项目结构

```
agi_agent/
├── agent.py              # 智能体核心类
├── run_agent.py          # 运行入口
├── diagnostics.py        # 诊断工具
├── requirements.txt      # 依赖列表
├── config/               # 配置模块
│   ├── __init__.py
│   └── settings.py       # 全局配置参数
├── perception/           # 感知系统
│   ├── __init__.py
│   ├── autoencoder.py    # 生长自编码器
│   └── multimodal_fusion.py  # 多模态融合
├── cognitive/            # 认知系统
│   ├── __init__.py
│   ├── dual_system.py    # 双系统认知
│   ├── inference_engine.py   # 推理引擎
│   ├── causal_reasoning.py   # 因果推理
│   └── spiking_nn.py     # 脉冲神经网络
├── memory/               # 记忆系统
│   ├── __init__.py
│   ├── memory_harness.py # 记忆管理器
│   ├── memory_store.py   # 存储实现
│   └── memory_tiers.py   # 记忆层级定义
├── learning/             # 学习模块
│   ├── __init__.py
│   ├── knowledge_graph.py    # 知识图谱
│   ├── knowledge_ingestor.py # 知识摄入
│   └── meta_learning.py      # 元学习
├── decision/             # 决策系统
│   ├── __init__.py
│   ├── decision_engine.py    # 决策引擎
│   ├── action_planner.py     # 行动规划
│   └── execution_monitor.py  # 执行监控
├── evolution/            # 进化引擎
│   ├── __init__.py
│   ├── quad_level_evolution.py   # 四级进化
│   ├── dual_loop_evolution.py    # 双循环进化
│   └── neat_engine.py      # NEAT进化
├── meta_cognitive/       # 元认知系统
│   ├── __init__.py
│   ├── meta_cognitive_orchestrator.py
│   ├── self_model.py      # 自我模型
│   └── strategy_regulator.py
├── self_improvement/     # 自我改进系统
│   ├── __init__.py
│   ├── self_improver.py   # 递归自我改进
│   ├── performance_evaluator.py
│   └── safety_verifier.py
├── security/             # 安全模块
│   ├── __init__.py
│   ├── safety_monitor.py  # 安全监控
│   ├── compliance_checker.py   # 合规检查
│   ├── hard_boundary.py   # 硬边界系统
│   └── risk_classifier.py # 风险分类
├── file_ingestion/       # 文件摄入系统
│   ├── __init__.py
│   ├── file_access.py     # 文件访问管理
│   ├── file_parsers.py    # 文件解析器
│   ├── preprocessor.py    # 数据预处理
│   ├── vectorization.py   # 特征向量化
│   ├── structured_storage.py   # 结构化存储
│   └── file_ingestor.py   # 摄入管理器
├── webui/                # WebUI界面
│   ├── __init__.py
│   ├── app.py             # FastAPI应用
│   ├── api_server.py      # API路由
│   ├── static/            # 静态资源
│   │   ├── index.html
│   │   ├── app.js
│   │   ├── style.css
│   │   ├── i18n.js
│   │   └── locales/       # 国际化资源
│   └── uploads/           # 文件上传目录（gitignored）
├── tests/                # 测试用例
│   ├── test_functional.py
│   ├── test_performance.py
│   ├── test_security.py
│   └── test_file_ingestion.py
└── docs/                 # 文档
    ├── technical_documentation.md
    └── user_manual.md
```

### 配置说明

配置文件位于 `agi_agent/config/settings.py`：

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

### 插件开发

参考 [PLUGIN_DEVELOPMENT_SPEC.md](agi_agent/plugins/PLUGIN_DEVELOPMENT_SPEC.md) 了解插件开发规范。

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

# 运行特定测试模块
python -m pytest agi_agent/tests/test_functional.py -v
python -m pytest agi_agent/tests/test_performance.py -v
python -m pytest agi_agent/tests/test_security.py -v
python -m pytest agi_agent/tests/test_file_ingestion.py -v
```

### 测试覆盖

| 测试模块 | 覆盖范围 |
|----------|----------|
| **test_functional.py** | 功能测试：感知、认知、学习、进化、执行、元认知、安全 |
| **test_performance.py** | 性能测试：延迟、吞吐量、内存稳定性 |
| **test_security.py** | 安全测试：安全监控、合规检查、边界保护 |
| **test_file_ingestion.py** | 文件摄入测试：文件访问、解析、预处理、向量化、存储 |

---

## 🛡️ 安全框架 / Security Framework

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

---

## 🤝 贡献指南 / Contributing

欢迎贡献代码！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 获取详细信息。

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📞 联系方式 / Contact

- 项目地址: https://github.com/taojio/agi-agent
- 问题反馈: https://github.com/taojio/agi-agent/issues

---

**Made with ❤️ by the AGI Agent Team**