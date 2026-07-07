"""
安装向导配置生成器
用于生成 Inno Setup 或 NSIS 安装脚本
"""

import os
from typing import Dict, Any, List


def create_installer_config(app_name: str = "AGI Agent",
                            app_version: str = "1.0.0",
                            app_publisher: str = "AGI Team",
                            exe_path: str = "dist/AGI_Agent.exe",
                            output_dir: str = "installer") -> Dict[str, Any]:
    config = {
        "app_name": app_name,
        "app_version": app_version,
        "app_publisher": app_publisher,
        "exe_path": exe_path,
        "output_dir": output_dir,
        "default_install_dir": r"{pf}\AGI Agent",
        "start_menu_folder": "AGI Agent",
        "create_desktop_icon": True,
        "create_start_menu_icon": True,
        "license_file": "LICENSE",
        "readme_file": "README.md",
        "required_dlls": [],
        "data_files": [],
    }
    return config


def generate_inno_setup_script(config: Dict[str, Any]) -> str:
    script = f"""[Setup]
AppName={config['app_name']}
AppVersion={config['app_version']}
AppPublisher={config['app_publisher']}
DefaultDirName={config['default_install_dir']}
DefaultGroupName={config['start_menu_folder']}
OutputDir={config['output_dir']}
OutputBaseFilename={config['app_name'].replace(' ', '_')}_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"; Flags: unchecked

[Files]
Source: "{config['exe_path']}"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{{app}}"; Flags: ignoreversion

[Icons]
Name: "{{group}}\\{config['app_name']}"; Filename: "{{app}}\\{os.path.basename(config['exe_path'])}"
Name: "{{group}}\\卸载 {config['app_name']}"; Filename: "{{uninstallexe}}"
Name: "{{autodesktop}}\\{config['app_name']}"; Filename: "{{app}}\\{os.path.basename(config['exe_path'])}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{os.path.basename(config['exe_path'])}"; Description: "启动 {config['app_name']}"; Flags: nowait postinstall skipifsilent
"""
    return script


def generate_nsis_script(config: Dict[str, Any]) -> str:
    script = f"""!define APPNAME "{config['app_name']}"
!define VERSION "{config['app_version']}"
!define PUBLISHER "{config['app_publisher']}"
!define EXENAME "{os.path.basename(config['exe_path'])}"

Name "${{APPNAME}}"
OutFile "{config['output_dir']}/${{APPNAME}}_Setup.exe"
InstallDir "$PROGRAMFILES\\${{APPNAME}}"
RequestExecutionLevel admin

Section "主程序" SecMain
    SetOutPath $INSTDIR
    File "{config['exe_path']}"
    File "LICENSE"
    File "README.md"

    CreateDirectory "$SMPROGRAMS\\${{APPNAME}}"
    CreateShortCut "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\${{EXENAME}}"
    CreateShortCut "$SMPROGRAMS\\${{APPNAME}}\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"

    CreateShortCut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\${{EXENAME}}"

    WriteUninstaller "$INSTDIR\\uninstall.exe"
SectionEnd

Section "卸载"
    Delete $INSTDIR\\${{EXENAME}}
    Delete $INSTDIR\\LICENSE
    Delete $INSTDIR\\README.md
    Delete $INSTDIR\\uninstall.exe

    Delete "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk"
    Delete "$SMPROGRAMS\\${{APPNAME}}\\Uninstall.lnk"
    RMDir "$SMPROGRAMS\\${{APPNAME}}"

    Delete "$DESKTOP\\${{APPNAME}}.lnk"

    RMDir $INSTDIR
SectionEnd
"""
    return script


def generate_build_script() -> str:
    script = r"""@echo off
echo ========================================
echo   AGI Agent - 构建打包脚本
echo ========================================
echo.

echo [1/3] 清理旧的构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [2/3] 使用 PyInstaller 打包...
pyinstaller --noconfirm build.spec

if errorlevel 1 (
    echo 打包失败!
    pause
    exit /b 1
)

echo [3/3] 构建完成!
echo.
echo 可执行文件位于: dist\AGI_Agent.exe
echo.
pause
"""
    return script


def save_build_scripts(output_dir: str = "./build_scripts"):
    os.makedirs(output_dir, exist_ok=True)

    config = create_installer_config()

    inno_path = os.path.join(output_dir, "setup.iss")
    with open(inno_path, "w", encoding="utf-8") as f:
        f.write(generate_inno_setup_script(config))

    nsis_path = os.path.join(output_dir, "setup.nsi")
    with open(nsis_path, "w", encoding="utf-8") as f:
        f.write(generate_nsis_script(config))

    bat_path = os.path.join(output_dir, "build.bat")
    with open(bat_path, "w", encoding="gbk") as f:
        f.write(generate_build_script())

    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""
AGI Agent 打包指南
==================

前置依赖:
1. Python 3.10+
2. PyInstaller (pip install pyinstaller)
3. 项目所有依赖已安装

快速打包:
1. 运行 build.bat
2. 等待打包完成
3. 在 dist/ 目录找到 AGI_Agent.exe

制作安装包:
- 使用 Inno Setup: 打开 setup.iss 编译
- 使用 NSIS: 编译 setup.nsi

系统要求:
- Windows 10/11 64位
- 4GB+ 内存
- 可选: NVIDIA GPU (CUDA)
""")

    return {
        "inno_setup": inno_path,
        "nsis": nsis_path,
        "build_bat": bat_path,
        "readme": readme_path
    }


if __name__ == "__main__":
    result = save_build_scripts()
    print("构建脚本已生成:")
    for name, path in result.items():
        print(f"  {name}: {path}")
