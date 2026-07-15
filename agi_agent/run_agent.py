import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.agent import SelfEvolvingAGI

if __name__ == "__main__":
    agi_agent = SelfEvolvingAGI(input_dim=512)
    logging.getLogger("agi_agent").info("===== Self-Evolving AGI Agent Started =====")
    logging.getLogger("agi_agent").info("Core capabilities: Meta-cognition | Meta-learning | Unsupervised adaptation")
    logging.getLogger("agi_agent").info("                  Autonomous thinking | Self-evolution | Autonomous action")
    
    report = agi_agent.run(steps=1000)
    logging.getLogger("agi_agent").info("\n===== Run Report =====")
    logging.getLogger("agi_agent").info(f"Final Step: {report['agent_info']['step']}")
    logging.getLogger("agi_agent").info(f"Performance Score: {report['performance']['performance_score']['total_score']:.4f}")
    logging.getLogger("agi_agent").info(f"Free Energy: {report['cognitive_metrics']['cognitive']['free_energy']:.4f}")
    logging.getLogger("agi_agent").info(f"Confidence: {report['cognitive_metrics']['cognitive']['confidence']:.4f}")
    logging.getLogger("agi_agent").info(f"Knowledge Rules: {report['knowledge']['count']}")
    logging.getLogger("agi_agent").info(f"Safety Risk Level: {report['safety']['risk_level']}")
    logging.getLogger("agi_agent").info(f"Compliance Rate: {report['compliance']['compliance_rate']:.2f}")
