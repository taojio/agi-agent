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


def main():
    multiprocessing.freeze_support()

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    if getattr(sys, 'frozen', False):
        os.chdir(base_dir)

    sys.path.insert(0, base_dir)

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
