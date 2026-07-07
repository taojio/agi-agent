# Self-Evolving AGI Agent - User Manual

## 1. Introduction

Welcome to the Self-Evolving AGI Agent! This autonomous AI system is designed to continuously learn, adapt, and evolve without external supervision. The agent implements cutting-edge AI techniques including predictive coding, free energy principle, and neuroevolution to achieve self-directed intelligence.

### What Can This Agent Do?

- **Autonomous Learning**: The agent learns patterns from its environment without explicit labels
- **Self-Optimization**: It automatically adjusts its neural network structure and learning strategies
- **Continuous Evolution**: The agent evolves its architecture over time to improve performance
- **Safety Awareness**: Built-in safety protocols ensure responsible operation
- **Performance Monitoring**: Real-time metrics and visualization tools

## 2. Installation

### 2.1 Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- At least 4GB RAM (8GB recommended)
- GPU with CUDA support (optional but recommended)

### 2.2 Install Dependencies

```bash
cd agi_agent
pip install -r requirements.txt
```

### 2.3 Verify Installation

```bash
python -c "from agent import SelfEvolvingAGI; print('Installation successful!')"
```

## 3. Quick Start

### 3.1 Basic Usage

```python
from agent import SelfEvolvingAGI

# Create agent with 16-dimensional input
agent = SelfEvolvingAGI(input_dim=16)

# Run autonomous evolution for 1000 steps
report = agent.run(steps=1000)

# View results
print(f"Final Step: {report['agent_info']['step']}")
print(f"Performance Score: {report['performance']['performance_score']['total_score']:.4f}")
print(f"Free Energy: {report['cognitive_metrics']['cognitive']['free_energy']:.4f}")
print(f"Confidence: {report['cognitive_metrics']['cognitive']['confidence']:.4f}")
print(f"Knowledge Rules: {report['knowledge']['count']}")
print(f"Safety Risk Level: {report['safety']['risk_level']}")
print(f"Compliance Rate: {report['compliance']['compliance_rate']:.2f}")
```

### 3.2 Interactive Mode

```python
from agent import SelfEvolvingAGI
import numpy as np

agent = SelfEvolvingAGI(input_dim=16)

# Simulate environment interaction
for step in range(100):
    # Generate random environment observation
    observation = np.random.uniform(-1, 1, 16)
    
    # Let the agent process and act
    result = agent.step(observation)
    
    # Print progress every 20 steps
    if step % 20 == 0:
        print(f"Step {step}:")
        print(f"  Free Energy: {result['free_energy']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Novelty: {result['novelty']:.4f}")
```

## 4. Configuration

### 4.1 Adjusting Parameters

Edit `config/settings.py` to customize agent behavior:

```python
# Core thresholds
FREE_ENERGY_THRESHOLD = 0.05    # Lower = more conservative learning
EVOLVE_TRIGGER_STEP = 200       # Steps before evolution can trigger
MAX_INFERENCE_STEP = 5          # How far ahead the agent predicts
NOVELTY_THRESHOLD = 0.12        # What counts as "new" environment

# Memory settings
MEMORY_BUFFER_SIZE = 200        # Size of experience buffer
KNOWLEDGE_MAX_SIZE = 1000       # Maximum knowledge rules

# Safety constraints
SAFETY_MAX_MEMORY_GB = 4.0      # Memory limit
SAFETY_MAX_LATENCY_MS = 1000    # Timeout threshold
```

### 4.2 Environment Integration

Provide custom environment observations:

```python
def custom_env():
    """Return observation from your environment"""
    # Replace with actual sensor data
    return np.random.uniform(-1, 1, 16)

agent = SelfEvolvingAGI(input_dim=16)
report = agent.run(steps=1000, env_generator=custom_env)
```

## 5. Understanding Agent Behavior

### 5.1 Key Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Free Energy** | Prediction error (lower = better) | 0-∞ |
| **Confidence** | Cognitive certainty | 0-1 |
| **Novelty** | How new the environment is | 0-1 |
| **Entropy** | Cognitive uncertainty | 0-∞ |
| **Latency** | Processing time (ms) | 0-∞ |

### 5.2 Agent States

- **healthy**: Normal operation
- **warning**: High free energy detected
- **critical**: Severe issues requiring attention

### 5.3 Evolution Triggers

The agent will evolve its structure when:
- Free energy remains high for extended periods
- Environment is stable (low novelty)
- Agent has accumulated enough experience

## 6. Advanced Features

### 6.1 Hardware Self-Expansion

The agent can adapt to new input dimensions:

```python
agent = SelfEvolvingAGI(input_dim=16)

# Simulate hardware upgrade
agent.hardware_self_expand(new_input_dim=24)

# Continue with new dimension
agent.run(steps=500)
```

### 6.2 Saving and Loading

```python
agent = SelfEvolvingAGI(input_dim=16)

# Run for a while
agent.run(steps=500)

# Save checkpoint
agent.save_checkpoint()

# Create new agent
new_agent = SelfEvolvingAGI(input_dim=16)

# Load from checkpoint
new_agent.load_checkpoint()

# Continue from where we left off
new_agent.run(steps=500)
```

### 6.3 Generating Reports

```python
agent = SelfEvolvingAGI(input_dim=16)
agent.run(steps=1000)

# Generate comprehensive report
report = agent.generate_report()

# Access different report sections
print("Agent Info:", report["agent_info"])
print("Cognitive Metrics:", report["cognitive_metrics"])
print("Performance:", report["performance"])
print("Safety:", report["safety"])
print("Compliance:", report["compliance"])
```

### 6.4 Visualization

The agent automatically generates visualizations:

1. **Metric Charts**: Saved to `visualizations/` directory
2. **Dashboard**: HTML file with key metrics
3. **Evolution Plots**: Fitness and speciation over time

## 7. Testing

### 7.1 Run All Tests

```bash
cd agi_agent
python -m pytest tests/ -v
```

### 7.2 Run Specific Tests

```bash
# Functional tests
python -m pytest tests/test_functional.py -v

# Performance tests
python -m pytest tests/test_performance.py -v

# Security tests
python -m pytest tests/test_security.py -v
```

## 8. Safety Considerations

### 8.1 Safety Protocols

The agent includes multiple safety layers:

1. **Resource Monitoring**: Tracks CPU, memory, and GPU usage
2. **Energy Constraints**: Limits maximum free energy
3. **Emergency Shutdown**: Triggers if critical thresholds are exceeded
4. **Compliance Checks**: Verifies bias, privacy, and transparency

### 8.2 Risk Levels

| Level | Description | Action |
|-------|-------------|--------|
| **low** | Normal operation | Continue |
| **medium** | Warning threshold exceeded | Monitor closely |
| **high** | Multiple warnings | Reduce computational load |
| **emergency** | Critical violation | Shutdown |

### 8.3 Custom Safety Rules

Edit `security/safety_monitor.py` to add custom safety rules:

```python
def check_custom_rule(self, metrics):
    """Add your custom safety check here"""
    if metrics["custom_metric"] > THRESHOLD:
        return {"type": "custom_violation", "severity": "warning"}
    return None
```

## 9. Troubleshooting

### 9.1 Common Issues

**Q: Agent is not learning (free energy stays high)**

A: Try increasing the learning rate or reducing the free energy threshold. Check if your input data has meaningful patterns.

**Q: Memory usage is increasing**

A: The agent has built-in memory limits. Check if `MEMORY_BUFFER_SIZE` is set appropriately.

**Q: Evolution is not triggering**

A: Evolution requires at least `EVOLVE_TRIGGER_STEP` steps and high free energy in a stable environment.

**Q: CUDA out of memory**

A: Reduce `hidden_dim` in the autoencoder or use CPU mode by setting `DEVICE = torch.device("cpu")`.

### 9.2 Logging

Logs are stored in the `logs/` directory. Check them for detailed error information.

## 10. FAQ

### Q: What is the free energy principle?

A: The free energy principle states that biological systems minimize their variational free energy, which is a measure of prediction error. This agent applies this principle to AI systems.

### Q: Does the agent require training data?

A: No, the agent is fully unsupervised. It learns from raw observations without labels or rewards.

### Q: Can I integrate this with real sensors?

A: Yes! Replace the random environment generator with your sensor data.

### Q: How does evolution work?

A: The agent uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve neural network structures over generations.

### Q: Is this truly autonomous?

A: Yes! Once initialized, the agent runs without any human intervention. It monitors itself, optimizes its learning, and evolves its structure automatically.

## 11. Support

For technical support, please check:
- Technical documentation: `docs/technical_documentation.md`
- Source code comments
- Test files for usage examples

## 12. License

This project is for research purposes. Please see the LICENSE file for details.