# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

BASE_DIR = Path(SPECPATH)

main_script = BASE_DIR / 'duplicate_link_tool_windows.py'
icon_path = BASE_DIR / 'assets' / 'duplicate_link_tool.ico'
version_info = BASE_DIR / 'version_info.txt'


a = Analysis(
    [str(main_script)],
    pathex=[str(BASE_DIR)],
    binaries=[],
    datas=[
        (str(icon_path), 'assets'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DuplicateLinkTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=False,
    manifest=str(BASE_DIR / 'assets' / 'dlt_manifest.xml'),
    icon=str(icon_path),
    version=str(version_info),
)
