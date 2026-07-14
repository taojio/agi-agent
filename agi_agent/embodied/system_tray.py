import sys
import os
import time
import threading
from typing import Callable, Dict, Any, Optional


class SystemTrayManager:
    def __init__(self, app_title: str = "AGI Agent"):
        self.app_title = app_title
        self._tray_icon = None
        self._running = False
        self._menu_items = []
        self._on_quit_callback: Optional[Callable] = None
        self._on_show_callback: Optional[Callable] = None

    def setup_tray(self, on_show: Callable = None, on_quit: Callable = None):
        self._on_show_callback = on_show
        self._on_quit_callback = on_quit

        try:
            import pystray
            from PIL import Image, ImageDraw

            image = Image.new('RGB', (64, 64), color=(70, 130, 180))
            dc = ImageDraw.Draw(image)
            dc.rectangle([20, 20, 44, 44], fill=(255, 255, 255))

            menu = pystray.Menu(
                pystray.MenuItem("显示面板", self._on_show),
                pystray.MenuItem("退出", self._on_quit),
            )

            self._tray_icon = pystray.Icon(
                "agi_agent",
                image,
                self.app_title,
                menu
            )

            return True
        except ImportError:
            return False

    def run(self):
        if self._tray_icon:
            self._running = True
            self._tray_icon.run()

    def run_detached(self):
        if self._tray_icon:
            thread = threading.Thread(target=self.run, daemon=True)
            thread.start()
            return thread
        return None

    def stop(self):
        self._running = False
        if self._tray_icon:
            self._tray_icon.stop()

    def _on_show(self, icon=None, item=None):
        if self._on_show_callback:
            try:
                self._on_show_callback()
            except Exception:
                pass

    def _on_quit(self, icon=None, item=None):
        if self._on_quit_callback:
            try:
                self._on_quit_callback()
            except Exception:
                pass
        self.stop()

    def update_tooltip(self, text: str):
        if self._tray_icon:
            self._tray_icon.title = text

    def notify(self, title: str, message: str):
        if self._tray_icon:
            try:
                self._tray_icon.notify(message, title)
            except Exception:
                pass


def run_embodied_app():
    import webbrowser

    print("正在初始化 AGI Agent...")

    tray = SystemTrayManager("AGI Agent")

    def on_show():
        webbrowser.open("http://localhost:8090")

    def on_quit():
        print("正在退出...")
        os._exit(0)

    tray_available = tray.setup_tray(on_show=on_show, on_quit=on_quit)

    if tray_available:
        tray.run_detached()
        print("系统托盘已启动，右键托盘图标可显示面板或退出")

    try:
        import uvicorn
        from agi_agent.webui.app import app

        print("=" * 60)
        print("  AGI Agent - 自进化智能体系统")
        print("=" * 60)
        print("  Web UI: http://localhost:8090")
        print("  按 Ctrl+C 停止")
        print("=" * 60)

        uvicorn.run(app, host="0.0.0.0", port=8090, log_level="warning")
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        tray.stop()


if __name__ == "__main__":
    run_embodied_app()
