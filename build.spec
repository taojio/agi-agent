# -*- mode: python ; coding: utf-8 -*-
"""
AGI Agent PyInstaller 打包配置
使用方法: pyinstaller build.spec
"""

import sys
import os
from pathlib import Path

block_cipher = None

project_root = Path('.').resolve()
agi_agent_dir = project_root / 'agi_agent'

hidden_imports = [
    'torch',
    'torch.nn',
    'torch.optim',
    'numpy',
    'fastapi',
    'uvicorn',
    'websockets',
    'pydantic',
    'starlette',
]

datas = [
    (str(agi_agent_dir / 'webui' / 'static'), 'static'),
    (str(agi_agent_dir / 'evolution' / 'neat_config.txt'), 'evolution'),
    (str(agi_agent_dir / 'docs'), 'docs'),
    (str(agi_agent_dir / 'plugins' / 'mods'), 'plugins/mods'),
]

binaries = []

excludes = [
    'tkinter',
    'matplotlib',
    'scipy',
    'pandas',
    'IPython',
    'notebook',
    'jupyter',
]

a = Analysis(
    [str(agi_agent_dir / 'run_agent.py')],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AGI_Agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
