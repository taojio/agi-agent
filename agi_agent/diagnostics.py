import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.agent import SelfEvolvingAGI


def run_diagnostics(steps=200):
    agent = SelfEvolvingAGI(input_dim=16)
    
    print("[DIAGNOSTICS] Starting module-by-module diagnostic...")
    print("=" * 80)
    
    fe_history = []
    confidence_history = []
    hidden_dim_history = []
    lr_history = []
    knowledge_rules_history = []
    memory_buffer_history = []
    
    for i in range(steps):
        obs = np.random.uniform(-1, 1, agent.input_dim)
        result = agent.step(obs)
        
        fe_history.append(result["free_energy"])
        confidence_history.append(result["confidence"])
        hidden_dim_history.append(agent.perception.hidden_dim)
        lr_history.append(agent.perception.optimizer.param_groups[0]['lr'])
        knowledge_rules_history.append(len(agent.cognitive.knowledge_rules))
        memory_buffer_history.append(len(agent.cognitive.memory_buffer))
        
        if (i + 1) % 50 == 0:
            print(f"\n[STEP {i+1}]")
            print(f"  Free Energy: {result['free_energy']:.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Hidden Dim: {agent.perception.hidden_dim}")
            print(f"  Learning Rate: {agent.perception.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Knowledge Rules: {len(agent.cognitive.knowledge_rules)}")
            print(f"  Memory Buffer: {len(agent.cognitive.memory_buffer)}")
            print(f"  System Status: {agent.meta_cog.system_status}")
    
    print("\n" + "=" * 80)
    print("[DIAGNOSTICS SUMMARY]")
    print("=" * 80)
    
    print("\n[PERCEPTION MODULE]")
    print(f"  Initial Hidden Dim: 32")
    print(f"  Final Hidden Dim: {hidden_dim_history[-1]}")
    print(f"  Hidden Dim Changes: {len(set(hidden_dim_history))} different values")
    print(f"  Avg Free Energy: {np.mean(fe_history):.4f}")
    print(f"  Min Free Energy: {np.min(fe_history):.4f}")
    print(f"  Max Free Energy: {np.max(fe_history):.4f}")
    
    print("\n[COGNITIVE MODULE]")
    print(f"  Avg Confidence: {np.mean(confidence_history):.4f}")
    print(f"  Max Confidence: {np.max(confidence_history):.4f}")
    print(f"  Knowledge Rules (final): {knowledge_rules_history[-1]}")
    print(f"  Memory Buffer (final): {memory_buffer_history[-1]}")
    
    print("\n[LEARNING MODULE]")
    print(f"  Learning Rates used: {set(lr_history)}")
    print(f"  Meta-learning exploration rate: {agent.meta_learn.exploration_rate:.4f}")
    
    print("\n[METACOGNITION MODULE]")
    print(f"  System Status: {agent.meta_cog.system_status}")
    trend = agent.meta_cog.get_trend_analysis()
    print(f"  FE Trend: {trend['fe_trend']:.4f}")
    print(f"  Novelty Trend: {trend['novelty_trend']:.4f}")
    
    print("\n[KNOWLEDGE GRAPH]")
    kg_summary = agent.knowledge_graph.get_summary()
    print(f"  Nodes: {kg_summary['nodes']}")
    print(f"  Edges: {kg_summary['edges']}")
    
    print("\n[SAFETY MODULE]")
    safety_report = agent.safety_monitor.get_safety_report()
    print(f"  Risk Level: {safety_report['risk_level']}")
    print(f"  Total Violations: {safety_report['total_violations']}")
    
    print("\n[CRITICAL ISSUES DETECTED]")
    issues = []
    
    if np.max(confidence_history) == 0.0:
        issues.append("✗ Confidence is always 0 - free energy never drops below threshold")
    
    if knowledge_rules_history[-1] == 0:
        issues.append("✗ Knowledge rules never deposited - fe never < FREE_ENERGY_THRESHOLD")
    
    if len(set(hidden_dim_history)) > 1:
        if np.mean(fe_history) > 0.1:
            issues.append("✗ Autoencoder keeps growing but free energy remains high")
    
    if safety_report['risk_level'] != 'low':
        issues.append(f"✗ Safety risk level is {safety_report['risk_level']}")
    
    if not issues:
        print("✓ No critical issues detected")
    else:
        for issue in issues:
            print(issue)
    
    print("\n" + "=" * 80)
    
    return {
        "fe_history": fe_history,
        "confidence_history": confidence_history,
        "hidden_dim_history": hidden_dim_history,
        "lr_history": lr_history,
        "knowledge_rules_history": knowledge_rules_history,
        "memory_buffer_history": memory_buffer_history,
        "issues": issues
    }


if __name__ == "__main__":
    run_diagnostics(steps=200)