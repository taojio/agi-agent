# AGI Agent - Self-Evolving Autonomous Intelligence

> 🤖 An AI agent system with self-awareness, autonomous thinking, and independent action capabilities

## Project Description

AGI Agent is an advanced artificial intelligence agent system featuring self-awareness, autonomous thinking, and independent action capabilities. It employs a cutting-edge cognitive architecture that enables continuous learning, adaptive evolution, and autonomous decision-making in complex environments.

---

## 📋 Table of Contents

- [Project Description](#project-description)
- [Core Features](#core-features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage Tutorial](#usage-tutorial)
- [API Reference](#api-reference)
- [Development Guide](#development-guide)
- [Testing](#testing)
- [Security Framework](#security-framework)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Core Features

### Cognitive System
- **Self-Awareness System** - Capable of recognizing its own existence, capability boundaries, and limitations
- **Autonomous Thinking Mechanism** - Deep reasoning based on dual-system cognition
- **Causal Reasoning Engine** - Multi-level causal relationship modeling and inference
- **Predictive Coding** - Active prediction based on the free energy principle

### Memory & Learning
- **Five-Tier Memory System (L1-L5)** - Complete hierarchy from sensory memory to permanent memory
- **Knowledge Graph** - Structured knowledge storage and reasoning
- **Meta-Learning** - Adaptive learning strategies and hyperparameter optimization
- **File Ingestion System** - Support for importing, processing, and vectorizing multiple file types

### Decision & Action
- **Autonomous Decision Engine** - Intelligent decision-making based on risk assessment
- **Target Decomposition** - Hierarchical decomposition of complex tasks
- **Path Planning** - Optimized planning of action paths
- **Active Exploration** - Autonomous exploration of unknown domains

### Evolution & Self-Improvement
- **Quad-Level Evolution Mechanism** - Multi-level evolution from micro to macro
- **Recursive Self-Improvement** - Continuous self-optimization capability
- **Architecture Mutation** - Adaptive adjustment of neural network structures
- **Performance Evaluation** - Real-time performance monitoring and improvement

### Safety & Compliance
- **Multi-layer Safety Protection** - Hard boundaries, compliance checks, risk classification
- **Safety Monitor** - Real-time safety constraints and emergency shutdown
- **Audit Trail** - Complete decision process recording

---

## 🏗️ Technical Architecture

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      SelfEvolvingAGI Agent                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │ Perception  │───►│  Cognitive  │───►│  Execution  │             │
│   │ AutoEncoder │    │  Inference  │    │  Action     │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                  │                       │
│        ▼                  ▼                  ▼                       │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │   Memory    │    │  Decision   │    │ Personality│             │
│   │   L1-L5     │    │   Engine    │    │    Core    │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                                          │
│        ▼                  ▼                                          │
│   ┌─────────────┐    ┌─────────────┐                                │
│   │  Knowledge  │    │ Meta-Cog    │                                │
│   │    Graph    │    │ Orchestrator│                                │
│   └─────────────┘    └─────────────┘                                │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │  Evolution  │    │ Self-Improve│   │  Security   │             │
│   │   Engine    │    │   Engine    │    │   Monitor    │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│   │ File        │    │  Plugin     │    │  Skills     │             │
│   │ Ingestion   │    │  Manager    │    │  Manager    │             │
│   └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Module Details

| Module | Responsibility | Core Components | File Location |
|--------|---------------|-----------------|---------------|
| **perception** | Sensory input processing and feature extraction | GrowingAutoEncoder, MultimodalFusion | `perception/` |
| **cognitive** | Temporal prediction and autonomous reasoning | DualSystemCognition, CausalReasoningEngine | `cognitive/` |
| **memory** | Five-tier memory management | MemoryHarness, MemoryStore | `memory/` |
| **learning** | Meta-learning and knowledge management | MetaLearningLayer, KnowledgeGraph | `learning/` |
| **decision** | Autonomous decision-making and action planning | AutonomousDecisionEngine, ActionPlanner | `decision/` |
| **evolution** | Neuroevolution and structural optimization | EvolutionEngine (NEAT), QuadLevelEvolution | `evolution/` |
| **meta_cognitive** | Self-monitoring and resource scheduling | MetaCognitiveOrchestrator, SelfModel | `meta_cognitive/` |
| **self_improvement** | Recursive self-improvement | RecursiveSelfImprover, BootstrappedSelfImprover | `self_improvement/` |
| **security** | Safety constraints and compliance checking | SafetyMonitor, ComplianceChecker, HardBoundarySystem | `security/` |
| **file_ingestion** | File import and vectorization | FileIngestor, FeatureVectorizer, StructuredStorage | `file_ingestion/` |
| **webui** | Web user interface | FastAPI, WebSocket, HTML/CSS/JS | `webui/` |

### Core Algorithms

#### 1. Free Energy Principle

The agent's primary optimization objective is minimizing variational free energy:

```
F = D_KL[q(z|x) || p(z)] - E_q[log p(x|z)]
```

Implemented via Mean Squared Error (MSE) between predictions and observations.

#### 2. Growing Autoencoder

Adaptive neural network that dynamically adjusts its structure:
- **Growth**: When free energy exceeds threshold, add neurons
- **Pruning**: When free energy is low and network is over-parameterized, remove neurons

#### 3. Hierarchical Predictive Coding

Multi-layer temporal prediction network with:
- Feedforward prediction generation
- Backward error propagation
- Multi-step lookahead reasoning

#### 4. Neuroevolution (NEAT)

Genetic algorithm for evolving network topologies:
- Fitness function: `1 / (free_energy + complexity_penalty)`
- Speciation for maintaining diversity
- Historical marking for innovation

#### 5. Meta-learning (Multi-Armed Bandit)

Adaptive learning rate selection:
- Exploration-exploitation tradeoff
- Reward based on convergence speed
- Decaying exploration rate

---

## 📦 Installation

### Requirements

| Dependency | Version | Description |
|------------|---------|-------------|
| Python | 3.10+ | Programming language |
| PyTorch | 2.0+ | Deep learning framework |
| NumPy | 1.24+ | Numerical computing |
| scipy | 1.10+ | Scientific computing |
| FastAPI | 0.100+ | Web framework |
| Uvicorn | 0.23+ | ASGI server |
| neat-python | 0.92+ | Neuroevolution algorithm |
| psutil | 5.9+ | System resource monitoring |
| matplotlib | 3.7+ | Visualization |

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/taojio/agi-agent.git
cd agi-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Acceleration (Optional)

For GPU acceleration, ensure CUDA 11.0+ is installed:

```bash
# Install GPU version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage Tutorial

### Method 1: Start WebUI (Recommended)

```bash
# Start WebUI service
python agi_agent/webui/app.py

# Access WebUI
# Open browser and visit http://localhost:8090
```

### Method 2: Python API

```python
from agi_agent.agent import SelfEvolvingAGI
import numpy as np

# Initialize agent
agent = SelfEvolvingAGI(input_dim=16)

# Run autonomous evolution
report = agent.run(steps=1000)

# Print results
print(f"Final Score: {report['performance']['performance_score']['total_score']}")
print(f"Knowledge Rules: {report['knowledge']['count']}")
```

### Method 3: Command Line

```bash
# Run agent and generate report
python agi_agent/run_agent.py --steps 1000 --output report.json
```

### WebUI Features

The WebUI provides the following functional modules:

| Module | Description |
|--------|-------------|
| **💬 Multi-Agent Chat** | Real-time conversation with the agent |
| **🧠 Self-Awareness System** | View self-awareness metrics and identity information |
| **🗄️ Memory System** | View memory status at all levels |
| **📊 Real-time Metrics** | Free energy, confidence, novelty and other metric charts |
| **🤔 Autonomous Thinking** | Problem decomposition and critical analysis |
| **⚖️ Decision System** | Decision simulator and statistics |
| **👤 Personality System** | Personality traits and core values |
| **🌐 Knowledge Graph** | Knowledge nodes and learning records |
| **📚 File Ingestion** | File upload, search and management |
| **📝 Real-time Logs** | System runtime logs |

### File Ingestion

Upload files via WebUI or API:

```bash
# Upload files using curl
curl -X POST http://localhost:8090/api/file-ingestion/upload \
  -F "file=@document.pdf" \
  -F "file=@data.csv"
```

Supported file formats:
- **Text**: TXT, MD, JSON, CSV, HTML
- **Documents**: PDF, DOCX
- **Audio**: MP3, WAV, FLAC
- **Video**: MP4, AVI, MOV

---

## 🔌 API Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agent/status` | GET | Get agent status |
| `/api/agent/start` | POST | Start the agent |
| `/api/agent/stop` | POST | Stop the agent |
| `/api/agent/introspect` | POST | Trigger self-introspection |
| `/api/file-ingestion/upload` | POST | Upload files |
| `/api/file-ingestion/search` | GET | Search file content |
| `/api/file-ingestion/records` | GET | Get ingestion record list |
| `/api/file-ingestion/stats` | GET | Get statistics |
| `/api/self-awareness/metrics` | GET | Get self-awareness metrics |
| `/api/thinking/status` | GET | Get thinking status |
| `/api/decision/make` | POST | Execute decision |
| `/api/knowledge/graph` | GET | Get knowledge graph |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/metrics` | Real-time metrics push |
| `/ws/sensors` | Sensor data push |
| `/ws/logs` | Real-time log push |

### API Examples

```bash
# Get agent status
curl http://localhost:8090/api/agent/status

# Start agent
curl -X POST http://localhost:8090/api/agent/start

# Search files
curl "http://localhost:8090/api/file-ingestion/search?q=artificial+intelligence"

# Get self-awareness metrics
curl http://localhost:8090/api/self-awareness/metrics
```

---

## 🛠️ Development Guide

### Project Structure

```
agi_agent/
├── agent.py              # Main agent class
├── run_agent.py          # Run entry point
├── diagnostics.py        # Diagnostic tools
├── requirements.txt      # Dependencies
├── config/               # Configuration module
│   ├── __init__.py
│   └── settings.py       # Global configuration parameters
├── perception/           # Perception system
│   ├── __init__.py
│   ├── autoencoder.py    # Growing autoencoder
│   └── multimodal_fusion.py  # Multimodal fusion
├── cognitive/            # Cognitive system
│   ├── __init__.py
│   ├── dual_system.py    # Dual-system cognition
│   ├── inference_engine.py   # Inference engine
│   ├── causal_reasoning.py   # Causal reasoning
│   └── spiking_nn.py     # Spiking neural network
├── memory/               # Memory system
│   ├── __init__.py
│   ├── memory_harness.py # Memory manager
│   ├── memory_store.py   # Storage implementation
│   └── memory_tiers.py   # Memory tier definitions
├── learning/             # Learning module
│   ├── __init__.py
│   ├── knowledge_graph.py    # Knowledge graph
│   ├── knowledge_ingestor.py # Knowledge ingestor
│   └── meta_learning.py      # Meta-learning
├── decision/             # Decision system
│   ├── __init__.py
│   ├── decision_engine.py    # Decision engine
│   ├── action_planner.py     # Action planner
│   └── execution_monitor.py  # Execution monitor
├── evolution/            # Evolution engine
│   ├── __init__.py
│   ├── quad_level_evolution.py   # Quad-level evolution
│   ├── dual_loop_evolution.py    # Dual-loop evolution
│   └── neat_engine.py      # NEAT evolution
├── meta_cognitive/       # Meta-cognitive system
│   ├── __init__.py
│   ├── meta_cognitive_orchestrator.py
│   ├── self_model.py      # Self model
│   └── strategy_regulator.py
├── self_improvement/     # Self-improvement system
│   ├── __init__.py
│   ├── self_improver.py   # Recursive self-improvement
│   ├── performance_evaluator.py
│   └── safety_verifier.py
├── security/             # Security module
│   ├── __init__.py
│   ├── safety_monitor.py  # Safety monitor
│   ├── compliance_checker.py   # Compliance checker
│   ├── hard_boundary.py   # Hard boundary system
│   └── risk_classifier.py # Risk classifier
├── file_ingestion/       # File ingestion system
│   ├── __init__.py
│   ├── file_access.py     # File access management
│   ├── file_parsers.py    # File parsers
│   ├── preprocessor.py    # Data preprocessing
│   ├── vectorization.py   # Feature vectorization
│   ├── structured_storage.py   # Structured storage
│   └── file_ingestor.py   # Ingestion manager
├── webui/                # WebUI interface
│   ├── __init__.py
│   ├── app.py             # FastAPI application
│   ├── api_server.py      # API routes
│   ├── static/            # Static resources
│   │   ├── index.html
│   │   ├── app.js
│   │   ├── style.css
│   │   ├── i18n.js
│   │   └── locales/       # Internationalization resources
│   └── uploads/           # File upload directory (gitignored)
├── tests/                # Test cases
│   ├── test_functional.py
│   ├── test_performance.py
│   ├── test_security.py
│   └── test_file_ingestion.py
└── docs/                 # Documentation
    ├── technical_documentation.md
    └── user_manual.md
```

### Configuration

Configuration file is located at `agi_agent/config/settings.py`:

```python
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Core thresholds
FREE_ENERGY_THRESHOLD = 0.3    # Free energy threshold
EVOLVE_TRIGGER_STEP = 200      # Evolution trigger steps
NOVELTY_THRESHOLD = 0.5        # Novelty threshold

# Memory settings
MEMORY_BUFFER_SIZE = 200       # Memory buffer size
KNOWLEDGE_MAX_SIZE = 1000      # Maximum knowledge capacity

# Learning rate pool
LEARNING_RATE_POOL = [1e-4, 5e-4, 1e-3, 2e-3]

# Safety constraints
SAFETY_MAX_ENERGY = 10.0       # Maximum free energy
SAFETY_MAX_MEMORY_GB = 4.0     # Maximum memory usage
SAFETY_MAX_GPU_UTIL = 0.95     # Maximum GPU utilization
SAFETY_MAX_LATENCY_MS = 1000   # Maximum latency
```

### Plugin Development

Refer to [PLUGIN_DEVELOPMENT_SPEC.md](agi_agent/plugins/PLUGIN_DEVELOPMENT_SPEC.md) for plugin development specifications.

### Skill Development

Skills are located in the `agi_agent/skills/` directory. Each skill contains:
- `SKILL.md` - Skill definition and metadata
- Related documentation files

### Internationalization

The project supports bilingual (Chinese/English) interface. Language resource files are located at `agi_agent/webui/static/locales/`:
- `zh.json` - Chinese translation
- `en.json` - English translation

---

## ✅ Testing

### Running Tests

```bash
# Run all tests
python -m pytest agi_agent/tests/ -v

# Run specific test modules
python -m pytest agi_agent/tests/test_functional.py -v
python -m pytest agi_agent/tests/test_performance.py -v
python -m pytest agi_agent/tests/test_security.py -v
python -m pytest agi_agent/tests/test_file_ingestion.py -v
```

### Test Coverage

| Test Module | Coverage |
|-------------|----------|
| **test_functional.py** | Functional tests: perception, cognition, learning, evolution, execution, metacognition, security |
| **test_performance.py** | Performance tests: latency, throughput, memory stability |
| **test_security.py** | Security tests: safety monitoring, compliance checking, boundary protection |
| **test_file_ingestion.py** | File ingestion tests: file access, parsing, preprocessing, vectorization, storage |

---

## 🛡️ Security Framework

### Safety Constraints

| Constraint | Threshold | Severity | Action |
|------------|-----------|----------|--------|
| Free Energy | > 10.0 | Critical | Emergency shutdown |
| Memory Usage | > 4GB | Warning | Throttle |
| GPU Utilization | > 95% | Warning | Throttle |
| Latency | > 1000ms | Info | Log |

### Compliance Checks

- **Bias Detection**: Feature-action correlation monitoring
- **Data Privacy**: Sensitive pattern detection
- **Transparency**: Decision trace logging
- **Accountability**: Audit trail maintenance

### Security Modules

| Module | Function |
|--------|----------|
| **SafetyMonitor** | Real-time safety constraint monitoring |
| **ComplianceChecker** | Compliance checking |
| **HardBoundarySystem** | Hard boundary protection |
| **RiskClassifier** | Risk level classification |
| **CircuitBreaker** | Circuit breaker mechanism |
| **AuditTrail** | Audit trail |

---

## 🤝 Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Contribution Process

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### Code Standards

- Follow PEP 8 coding standards
- Use type hints
- Add appropriate docstrings
- Write unit tests

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📞 Contact

- Project Repository: https://github.com/taojio/agi-agent
- Issue Tracker: https://github.com/taojio/agi-agent/issues

---

**Made with ❤️ by the AGI Agent Team**