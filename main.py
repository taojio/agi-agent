"""
AGI Agent 启动入口
用于打包为exe的主程序
"""

import sys
import os
import multiprocessing
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("agi_agent_main")

try:
    import torch
except ImportError:
    fallback_paths = [
        "D:\\Program Files\\Python311\\Lib\\site-packages",
        "C:\\Users\\Administrator\\python-sdk\\python3.10.16\\Lib\\site-packages",
        "D:\\torch_install",
    ]
    torch_found = False
    for path in fallback_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                import torch
                torch_found = True
                logger.info(f"使用备用路径加载 torch: {path}")
                break
            except ImportError:
                sys.path.remove(path)
    if not torch_found:
        raise ImportError("无法找到 torch 模块，请确保已安装 PyTorch")


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))


def handle_unhandled_rejection(promise, reason):
    logger.error(f"未处理的 Promise 拒绝: {reason}")


sys.excepthook = handle_uncaught_exception


def run_training_mode(total_steps=1000, input_dim=16):
    """运行训练模式 - 基于专属训练制度的智能体训练"""
    import numpy as np
    from agi_agent.agent import SelfEvolvingAGI

    logger.info("=" * 60)
    logger.info("  AGI Agent - 训练模式")
    logger.info("=" * 60)
    logger.info(f"  训练步数: {total_steps}")
    logger.info(f"  输入维度: {input_dim}")
    logger.info("=" * 60)

    agent = SelfEvolvingAGI(input_dim=input_dim)

    logger.info("开始训练...")
    for step in range(1, total_steps + 1):
        obs = np.random.randn(input_dim).astype(np.float32) * 0.1
        result = agent.step(obs)

        if step % 100 == 0:
            training_info = result.get("training_regime", {})
            logger.info(
                f"Step {step}/{total_steps} | "
                f"FE: {result.get('free_energy', 0):.4f} | "
                f"Conf: {result.get('confidence', 0):.4f} | "
                f"Phase: {training_info.get('phase_name', 'N/A')} | "
                f"Score: {training_info.get('overall_score', 0):.4f}"
            )

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)

    summary = agent.get_training_summary()
    logger.info(f"总步数: {summary['stats']['total_steps']}")
    logger.info(f"完成阶段数: {summary['stats']['phases_completed']}")
    logger.info(f"达成目标数: {summary['stats']['goals_achieved']}")
    logger.info(f"保存检查点数: {summary['stats']['checkpoints_saved']}")
    logger.info(f"架构变更数: {summary['stats']['architecture_changes']}")
    logger.info(f"整体目标进度: {summary['goals']['overall_progress']:.2%}")

    return agent


def main():
    multiprocessing.freeze_support()

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    if getattr(sys, 'frozen', False):
        os.chdir(base_dir)

    sys.path.insert(0, base_dir)

    # 解析命令行参数
    args = sys.argv[1:]
    mode = "webui"
    training_steps = 1000
    input_dim = 16

    i = 0
    while i < len(args):
        if args[i] in ("--train", "-t"):
            mode = "training"
        elif args[i] in ("--steps", "-s") and i + 1 < len(args):
            training_steps = int(args[i + 1])
            i += 1
        elif args[i] in ("--input-dim", "-d") and i + 1 < len(args):
            input_dim = int(args[i + 1])
            i += 1
        elif args[i] in ("--help", "-h"):
            print("用法: python main.py [选项]")
            print()
            print("选项:")
            print("  -t, --train          运行训练模式")
            print("  -s, --steps N        训练步数 (默认: 1000)")
            print("  -d, --input-dim D    输入维度 (默认: 16)")
            print("  -h, --help           显示帮助信息")
            print()
            print("示例:")
            print("  python main.py                # 启动 Web UI")
            print("  python main.py --train        # 运行训练模式")
            print("  python main.py -t -s 5000     # 训练5000步")
            return
        i += 1

    if mode == "training":
        run_training_mode(total_steps=training_steps, input_dim=input_dim)
        return

    try:
        from agi_agent.embodied.app import run_embodied_app
        run_embodied_app()
    except ImportError:
        try:
            import uvicorn
            from agi_agent.webui.app import app

            logger.info("=" * 60)
            logger.info("  AGI Agent - 自进化智能体系统")
            logger.info("=" * 60)
            logger.info(f"  工作目录: {os.getcwd()}")
            logger.info("  正在启动 Web UI...")
            logger.info("  访问地址: http://localhost:8090")
            logger.info("  按 Ctrl+C 停止")
            logger.info("=" * 60)

            uvicorn.run(app, host="0.0.0.0", port=8090, log_level="warning")
        except KeyboardInterrupt:
            logger.info("\n正在停止...")
        except Exception as e:
            logger.error(f"启动错误: {e}", exc_info=True)
            input("按回车键退出...")


if __name__ == "__main__":
    main()
