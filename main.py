"""
AGI Agent 启动入口
用于打包为exe的主程序
"""

import sys
import os
import multiprocessing

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
                print(f"[INFO] 使用备用路径加载 torch: {path}")
                break
            except ImportError:
                sys.path.remove(path)
    if not torch_found:
        raise ImportError("无法找到 torch 模块，请确保已安装 PyTorch")


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

            print("=" * 60)
            print("  AGI Agent - 自进化智能体系统")
            print("=" * 60)
            print(f"  工作目录: {os.getcwd()}")
            print("  正在启动 Web UI...")
            print("  访问地址: http://localhost:8090")
            print("  按 Ctrl+C 停止")
            print("=" * 60)

            uvicorn.run(app, host="0.0.0.0", port=8090, log_level="warning")
        except KeyboardInterrupt:
            print("\n正在停止...")
        except Exception as e:
            print(f"启动错误: {e}")
            input("按回车键退出...")


if __name__ == "__main__":
    main()
