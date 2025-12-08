import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional, Set, Tuple


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def cleanup_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)


def _ps_escape(value: str) -> str:
    return value.replace("'", "''")


def resolve_shortcut(path: Path) -> Tuple[bool, Optional[Path], Optional[str]]:
    """ショートカットのリンク先を解決する。
    
    Windowsのショートカット追跡機能を使用して、移動されたリンク先を自動的に検索・更新します。
    """
    # SLR_UPDATE (0x4) + SLR_NO_UI (0x1) = リンク先を検索して更新、UIなし
    command = (
        "$shell = New-Object -ComObject WScript.Shell;"
        f"$lnk = $shell.CreateShortcut('{_ps_escape(str(path))}');"
        "if ($lnk -and $lnk.TargetPath) {"
        "  $target = $lnk.TargetPath;"
        "  if (-not (Test-Path $target)) {"
        "    try {"
        "      $shellApp = New-Object -ComObject Shell.Application;"
        "      $folder = $shellApp.NameSpace((Split-Path $lnk.FullName -Parent));"
        "      $item = $folder.ParseName((Split-Path $lnk.FullName -Leaf));"
        "      if ($item) {"
        "        $link = $item.GetLink;"
        "        if ($link) {"
        "          $link.Resolve(5);"  # SLR_UPDATE | SLR_NO_UI
        "          $target = $link.Path;"
        "          if ($target -and (Test-Path $target)) {"
        "            $lnk.TargetPath = $target;"
        "            $lnk.Save();"
        "          }"
        "        }"
        "      }"
        "    } catch { }"
        "  }"
        "  Write-Output $target"
        "}"
    )
    try:
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False, None, 'PowerShell が見つかりません'

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or 'PowerShell コマンドが失敗しました'
        return False, None, message

    target_text = result.stdout.strip()
    if not target_text:
        return False, None, 'ショートカットのリンク先を取得できませんでした'

    return True, Path(target_text), None


def _build_realization_plan(src_root: Path) -> Tuple[List[Tuple[Path, Path]], Set[Path], List[str]]:
    files: List[Tuple[Path, Path]] = []
    dirs: Set[Path] = {Path('.')}
    warnings: List[str] = []

    def visit(path: Path, rel_path: Path, ancestors: Set[Path]):
        try:
            if path.is_dir():
                real_dir = path.resolve()
                dirs.add(rel_path)
                if real_dir in ancestors:
                    warnings.append(f'循環参照を検出: {path}')
                    return

                try:
                    entries = list(path.iterdir())
                except PermissionError as e:
                    warnings.append(f'アクセス拒否: {path} ({e})')
                    return

                new_ancestors = set(ancestors)
                new_ancestors.add(real_dir)
                for entry in entries:
                    entry_rel = rel_path / entry.name
                    if entry.suffix.lower() == '.lnk':
                        ok, target, message = resolve_shortcut(entry)
                        if not ok or target is None:
                            warnings.append(f'ショートカットを解決できません: {entry} ({message})')
                            continue
                        if not target.exists():
                            warnings.append(f'ショートカットのリンク先が存在しません: {entry} -> {target}')
                            continue
                        resolved_rel = entry_rel.with_suffix('')
                        if target.is_dir():
                            visit(target, resolved_rel, new_ancestors)
                        else:
                            files.append((target, resolved_rel))
                        continue
                    if entry.is_symlink():
                        try:
                            target = entry.resolve(strict=False)
                        except Exception as e:
                            warnings.append(f'リンク解決失敗: {entry} ({e})')
                            continue
                        if target.exists():
                            visit(target, entry_rel, new_ancestors)
                        else:
                            warnings.append(f'リンク先が存在しません: {entry}')
                        continue

                    visit(entry, entry_rel, new_ancestors)
            else:
                files.append((path, rel_path))
        except Exception as e:
            warnings.append(f'{path}: {e}')

    visit(src_root, Path('.'), set())
    return files, dirs, warnings


def realize_directory(src_root: Path, dst_root: Path, progress_cb=None) -> Tuple[int, int, List[str]]:
    if not src_root.exists():
        raise FileNotFoundError(f'ソースが存在しません: {src_root}')

    files, dirs, warnings = _build_realization_plan(src_root)

    ensure_dir(dst_root)
    for rel_dir in sorted(dirs, key=lambda p: len(p.parts)):
        if rel_dir == Path('.'):
            continue
        ensure_dir(dst_root / rel_dir)

    total = len(files)
    copied = 0
    if total == 0 and progress_cb:
        progress_cb(0, 0)

    for src_file, rel_path in files:
        dest_file = dst_root / rel_path
        ensure_dir(dest_file.parent)
        shutil.copy2(src_file, dest_file)
        copied += 1
        if progress_cb:
            progress_cb(copied, total)

    return copied, total, warnings


def zip_directory(src_root: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_root):
            rel_root = Path(root).relative_to(src_root)
            for d in dirs:
                arc_dir = (rel_root / d).as_posix() + '/'
                zf.writestr(arc_dir, '')
            for f in files:
                arcname = (rel_root / f).as_posix()
                file_path = Path(root) / f
                zf.write(file_path, arcname)

