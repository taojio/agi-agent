import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class MetricsVisualizer:
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_metrics(self, history: list, title: str, filename: str, ylabel: str = "Value"):
        if len(history) < 2:
            return None
        
        steps = [h["step"] for h in history]
        values = [h["free_energy"] if "free_energy" in h else h.get("value", 0.0) for h in history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, values, label=ylabel, color='#1f77b4')
        
        window = min(20, len(values))
        if window > 1:
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            plt.plot(smoothed_steps, smoothed, label=f"{ylabel} (smoothed)", color='#ff7f0e', linestyle='--')
        
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_multiple_metrics(self, history: list, title: str, filename: str):
        if len(history) < 2:
            return None
        
        steps = [h["step"] for h in history]
        
        plt.figure(figsize=(12, 8))
        
        metrics_to_plot = [
            ("free_energy", "Free Energy", '#1f77b4'),
            ("confidence", "Confidence", '#2ca02c'),
            ("novelty", "Novelty", '#ff7f0e')
        ]
        
        for key, label, color in metrics_to_plot:
            values = [h.get(key, 0.0) for h in history]
            plt.plot(steps, values, label=label, color=color, alpha=0.8)
            
            window = min(20, len(values))
            if window > 1:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
                plt.plot(smoothed_steps, smoothed, color=color, linestyle='--', alpha=0.5)
        
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_heatmap(self, data: np.ndarray, title: str, filename: str, xlabel: str = "", ylabel: str = ""):
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        filepath = os.path.join(self.output_dir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_dashboard(self, evaluation_report: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AGI Agent Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px; }}
                .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .score {{ font-size: 36px; font-weight: bold; color: #1f77b4; }}
                .label {{ color: #666; font-size: 14px; }}
                .status {{ padding: 5px 10px; border-radius: 20px; font-weight: bold; }}
                .status.healthy {{ background: #98fb98; color: #2e8b57; }}
                .status.warning {{ background: #ffdeb5; color: #cd853f; }}
                .status.critical {{ background: #ffb6c1; color: #dc143c; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>AGI Agent Performance Dashboard</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="grid">
                    <div class="metric-card">
                        <div class="score">{evaluation_report['performance_score']['total_score']:.4f}</div>
                        <div class="label">Overall Performance Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="score">{evaluation_report['performance_score']['averages']['free_energy']:.4f}</div>
                        <div class="label">Average Free Energy</div>
                    </div>
                    <div class="metric-card">
                        <div class="score">{evaluation_report['performance_score']['averages']['confidence']:.4f}</div>
                        <div class="label">Average Confidence</div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Convergence Status</h3>
                    <p>Converged: {evaluation_report['convergence']['converged']}</p>
                    <p>Improvement Rate: {evaluation_report['convergence']['rate']:.4f}</p>
                    <p>Recent Avg FE: {evaluation_report['convergence']['recent_avg_fe']:.4f}</p>
                </div>
                
                <div class="metric-card">
                    <h3>System Status</h3>
                    <span class="status healthy">HEALTHY</span>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return dashboard_path