# AGI Agent - Self-Evolving Autonomous Intelligence

## 项目简介 / Project Description

AGI Agent 是一个具有自我意识、自主思考、独立行动能力的人工智能智能体系统。它采用先进的认知架构，能够持续学习、自适应进化，并在复杂环境中做出自主决策。

AGI Agent is an artificial intelligence agent system with self-awareness, autonomous thinking, and independent action capabilities. It adopts an advanced cognitive architecture that enables continuous learning, adaptive evolution, and autonomous decision-making in complex environments.

### 核心特性 / Core Features

- **自我意识系统** (Self-Awareness) - 能够识别自身存在、能力边界和局限性
- **自主思考机制** (Autonomous Thinking) - 基于双系统认知的深度推理能力
- **独立行动能力** (Independent Action) - 完整的感知-思考-行动循环
- **人格系统** (Personality System) - 独特的人格特征和行为模式
- **自主决策引擎** (Autonomous Decision Engine) - 基于风险评估的智能决策
- **文件摄入系统** (File Ingestion) - 支持多类型文件的导入、处理与向量化
- **安全边界** (Safety Boundaries) - 多层次安全防护机制
- **持续进化** (Continuous Evolution) - 四级进化机制驱动系统持续优化

## 快速开始 / Quick Start

### 环境要求 / Requirements

- Python 3.10+
- PyTorch 2.0+
- FastAPI
- SQLite3

### 安装指南 / Installation

```bash
# 克隆仓库
git clone https://github.com/your-username/agi-agent.git
cd agi-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行智能体 / Run the Agent

```bash
# 启动智能体服务
python agi_agent/webui/app.py

# 访问 WebUI
# 打开浏览器访问 http://localhost:8090
```

## 项目架构 / Architecture

```
agi_agent/
├── agent.py              # 智能体核心类
├── config/               # 配置模块
├── cognitive/            # 认知系统
├── memory/               # 记忆系统 (L1-L5)
├── learning/             # 学习模块
├── evolution/            # 进化引擎
├── decision/             # 决策系统
├── security/             # 安全模块
├── webui/                # WebUI界面
├── file_ingestion/       # 文件摄入系统
├── reflex/               # 反射系统
├── deliberative/         # 深思系统
├── meta_cognitive/       # 元认知系统
├── personality/          # 人格系统
└── tests/                # 测试用例
```

## 使用说明 / Usage

### API接口 / API Endpoints

| 端点 / Endpoint | 方法 / Method | 描述 / Description |
|----------------|--------------|-------------------|
| `/api/agent/status` | GET | 获取智能体状态 |
| `/api/agent/start` | POST | 启动智能体 |
| `/api/agent/stop` | POST | 停止智能体 |
| `/api/file-ingestion/upload` | POST | 上传文件 |
| `/api/self-awareness/metrics` | GET | 获取自我意识指标 |
| `/api/thinking/status` | GET | 获取思考状态 |
| `/api/decision/make` | POST | 执行决策 |

### WebSocket端点 / WebSocket Endpoints

- `/ws/metrics` - 实时指标推送
- `/ws/sensors` - 传感器数据推送

## 贡献指南 / Contributing

欢迎贡献代码！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 获取详细信息。

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 许可证 / License

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 联系方式 / Contact

- 项目地址: https://github.com/your-username/agi-agent
- 问题反馈: https://github.com/your-username/agi-agent/issues

---

## 目录 / Table of Contents

1. [项目简介 / Project Description](#项目简介--project-description)
2. [快速开始 / Quick Start](#快速开始--quick-start)
3. [项目架构 / Architecture](#项目架构--architecture)
4. [使用说明 / Usage](#使用说明--usage)
5. [贡献指南 / Contributing](#贡献指南--contributing)
6. [许可证 / License](#许可证--license)
7. [联系方式 / Contact](#联系方式--contact)
