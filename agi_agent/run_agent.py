import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.agent import SelfEvolvingAGI

if __name__ == "__main__":
    agi_agent = SelfEvolvingAGI(input_dim=16)
    print("===== Self-Evolving AGI Agent Started =====")
    print("Core capabilities: Meta-cognition | Meta-learning | Unsupervised adaptation")
    print("                  Autonomous thinking | Self-evolution | Autonomous action")
    
    report = agi_agent.run(steps=1000)
    print("\n===== Run Report =====")
    print(f"Final Step: {report['agent_info']['step']}")
    print(f"Performance Score: {report['performance']['performance_score']['total_score']:.4f}")
    print(f"Free Energy: {report['cognitive_metrics']['cognitive']['free_energy']:.4f}")
    print(f"Confidence: {report['cognitive_metrics']['cognitive']['confidence']:.4f}")
    print(f"Knowledge Rules: {report['knowledge']['count']}")
    print(f"Safety Risk Level: {report['safety']['risk_level']}")
    print(f"Compliance Rate: {report['compliance']['compliance_rate']:.2f}")