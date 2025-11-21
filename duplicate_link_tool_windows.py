import os
import sys
import hashlib
import shutil
import stat
import time
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import ctypes
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import threading
import datetime
import traceback
from typing import Optional

from zip_util import cleanup_directory, realize_directory, resolve_shortcut, zip_directory


class HistoryEntry(tk.Entry):
    """履歴機能・Undo/RedoをサポートするEntry"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.history = []  # 入力履歴（確定時に追加）
        self.history_index = -1  # 現在の履歴位置
        self.undo_stack = []  # Undo用スタック
        self.redo_stack = []  # Redo用スタック
        self.last_content = self.get()
        
        # キーバインド
        self.bind('<Up>', self._on_up_key)
        self.bind('<Down>', self._on_down_key)
        self.bind('<Control-z>', self._on_undo)
        self.bind('<Control-y>', self._on_redo)
        self.bind('<Return>', self._on_enter)
        self.bind('<FocusOut>', self._save_to_history)
        self.bind('<KeyRelease>', self._on_key_release)
    
    def _on_key_release(self, event):
        """キー入力後にUndoスタックを更新"""
        # 制御キーは無視
        if event.keysym in ('Up', 'Down', 'Control_L', 'Control_R', 'Shift_L', 'Shift_R'):
            return
        
        current = self.get()
        if current != self.last_content:
            self.undo_stack.append(self.last_content)
            self.redo_stack.clear()
            self.last_content = current
            # スタックサイズ制限
            if len(self.undo_stack) > 100:
                self.undo_stack.pop(0)
    
    def _on_undo(self, event):
        """Ctrl+Z: 元に戻す"""
        if self.undo_stack:
            self.redo_stack.append(self.get())
            previous = self.undo_stack.pop()
            self.delete(0, tk.END)
            self.insert(0, previous)
            self.last_content = previous
        return 'break'
    
    def _on_redo(self, event):
        """Ctrl+Y: やり直し"""
        if self.redo_stack:
            self.undo_stack.append(self.get())
            next_content = self.redo_stack.pop()
            self.delete(0, tk.END)
            self.insert(0, next_content)
            self.last_content = next_content
        return 'break'
    
    def _save_to_history(self, event=None):
        """現在の内容を履歴に保存"""
        current = self.get().strip()
        if current and (not self.history or current != self.history[-1]):
            self.history.append(current)
            if len(self.history) > 50:  # 履歴の上限
                self.history.pop(0)
        self.history_index = len(self.history)
    
    def _on_enter(self, event):
        """Enterキー押下時に履歴に保存"""
        self._save_to_history()
        return None
    
    def _on_up_key(self, event):
        """上矢印: 古い履歴へ移動"""
        if not self.history:
            return 'break'
        
        if self.history_index > 0:
            self.history_index -= 1
            self.delete(0, tk.END)
            self.insert(0, self.history[self.history_index])
            self.last_content = self.get()
        return 'break'
    
    def _on_down_key(self, event):
        """下矢印: 新しい履歴へ移動"""
        if not self.history:
            return 'break'
        
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.delete(0, tk.END)
            self.insert(0, self.history[self.history_index])
            self.last_content = self.get()
        elif self.history_index == len(self.history) - 1:
            self.history_index = len(self.history)
            self.delete(0, tk.END)
            self.last_content = ''
        return 'break'


def _log_startup(message: str) -> None:
    """Record early-start events for troubleshooting elevation and packaging issues."""
    try:
        base = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
        base.mkdir(parents=True, exist_ok=True)
        with open(base / 'startup_debug.log', 'a', encoding='utf-8') as fp:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fp.write(f'[{timestamp}] {message}\n')
    except Exception:
        pass

# ---- 管理者権限での自動再起動 ----
def run_as_admin():
    if os.environ.get('DLTOOL_SKIP_ELEVATION') == '1':
        _log_startup('Elevation skipped via DLTOOL_SKIP_ELEVATION')
        return True
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            _log_startup('Already running with administrative privileges')
            return True

        exe_path = Path(sys.executable).resolve()
        params = ' '.join(f'"{arg}"' for arg in (sys.argv[1:] if getattr(sys, 'frozen', False) else sys.argv))
        _log_startup(f'Requesting elevation via ShellExecuteW: exe="{exe_path}" params="{params}"')
        result = ctypes.windll.shell32.ShellExecuteW(None, "runas", str(exe_path), params or None, None, 1)
        _log_startup(f'ShellExecuteW returned {result}')
        if result <= 32:
            _log_startup('Elevation request failed; terminating without relaunch')
            return False
        return None
    except Exception as exc:
        _log_startup(f'run_as_admin error: {exc!r}')
        return False


_admin_state = run_as_admin()
if _admin_state is True:
    pass
elif _admin_state is None:
    sys.exit(0)
else:
    ctypes.windll.user32.MessageBoxW(None, '管理者権限の取得に失敗したため終了します。', 'Duplicate Link Tool', 0x00000010)
    sys.exit(1)

# ---- 設定 ----
DARK_BG = '#1e1e1e'
DARK_FG = '#f0f0f0'
ACCENT = '#4a90e2'
CHUNK = 65536
THREADS = max(1, (os.cpu_count() or 2) - 1)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
ASSET_ICON_NAME = 'assets/duplicate_link_tool.ico'


def _resource_path(relative: str) -> Path:
    try:
        if getattr(sys, 'frozen', False):
            base = Path(getattr(sys, '_MEIPASS', BASE_DIR))
        else:
            base = BASE_DIR
    except Exception:
        base = BASE_DIR
    return base / relative

FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
CREATE_NO_WINDOW = 0x08000000


def _powershell_escape(value: str) -> str:
    return value.replace("'", "''")


def _build_shortcut_batch_script(entries: list[tuple[str, str]]) -> str:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$shell = New-Object -ComObject WScript.Shell",
        "try {",
    ]
    for link, target in entries:
        escaped_link = _powershell_escape(link)
        escaped_target = _powershell_escape(target)
        
        # ターゲットが相対パスかどうか判定
        is_relative = not Path(target).is_absolute()
        
        lines.append(f"    $lnk = $shell.CreateShortcut('{escaped_link}')")
        
        if is_relative:
            # 相対パスの場合：WorkingDirectoryを設定し、TargetPathは絶対パスに変換
            link_dir = str(Path(link).parent.resolve())
            target_abs = str((Path(link).parent / target).resolve())
            escaped_link_dir = _powershell_escape(link_dir)
            escaped_target_abs = _powershell_escape(target_abs)
            lines.append(f"    $lnk.WorkingDirectory = '{escaped_link_dir}'")
            lines.append(f"    $lnk.TargetPath = '{escaped_target_abs}'")
        else:
            # 絶対パスの場合：従来通り
            lines.append(f"    $lnk.TargetPath = '{escaped_target}'")
        
        lines.append("    $lnk.Save()")
    lines.extend(
        [
            "}",
            "finally {",
            "    if ($null -ne $shell) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($shell) | Out-Null }",
            "}",
        ]
    )
    return '\n'.join(lines)


def _run_shortcut_batch(entries: list[tuple[str, str]], *, script: Optional[str] = None) -> tuple[bool, str]:
    if not entries:
        return True, ''

    if script is None:
        script = _build_shortcut_batch_script(entries)

    temp_script_path: Optional[str] = None
    try:
        # UTF-8 BOMを使用してPowerShellが正しく認識できるようにする
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.ps1', encoding='utf-8-sig') as tmp:
            tmp.write(script)
            tmp.write('\n')
            temp_script_path = tmp.name

        result = subprocess.run(
            [
                'powershell',
                '-NoProfile',
                '-NonInteractive',
                '-ExecutionPolicy',
                'Bypass',
                '-WindowStyle',
                'Hidden',
                '-File',
                temp_script_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',  # 出力もUTF-8として扱う
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError:
        return False, 'powershell_not_found'
    except Exception as exc:
        return False, str(exc)
    finally:
        if temp_script_path:
            try:
                os.remove(temp_script_path)
            except OSError:
                pass

    ok = result.returncode == 0
    msg = result.stderr.strip() or result.stdout.strip()
    return ok, msg

# ---- ユーティリティ ----
def compute_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(CHUNK), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def format_size(num_bytes: int) -> str:
    """Return a human-readable size string (e.g., 12.4 MB)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"

# ---- 重複検出 ----
def collect_files(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            out.append(os.path.join(dirpath, fn))
    return out

def find_duplicates(
    main_dir,
    compare_dir,
    progress,
    tree,
    count_label,
    size_var,
    status_var,
    compare_nondup_container,
    main_nondup_container,
    open_compare_button,
    open_main_button,
    cancel_flag,
    tk_root,
    log_callback=None,
):
    def normalize(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))
    
    # 同じフォルダ比較かどうかを判定
    is_same_folder = normalize(main_dir) == normalize(compare_dir)
    if is_same_folder:
        if log_callback:
            log_callback('同じフォルダ内での重複検出を開始します（同一ファイルパスの組み合わせは除外）')

    main_files = collect_files(main_dir)
    
    main_lookup: dict[tuple[str, int], list[str]] = defaultdict(list)
    main_size_index: dict[int, list[str]] = defaultdict(list)
    main_size_cache: dict[str, int] = {}

    for p in main_files:
        try:
            size = os.path.getsize(p)
        except Exception:
            continue
        key = (os.path.basename(p), size)
        main_lookup[key].append(p)
        main_size_index[size].append(p)
        main_size_cache[p] = size

    main_hash_cache: dict[str, Optional[str]] = {}
    hash_lock = threading.Lock()

    def get_main_hash(path: str) -> Optional[str]:
        with hash_lock:
            cached = main_hash_cache.get(path)
        if cached is not None:
            return cached
        hashed = compute_hash(path)
        with hash_lock:
            main_hash_cache[path] = hashed
        return hashed

    compare_list = collect_files(compare_dir)
    
    if status_var is not None:
        try:
            tk_root.after(0, lambda: status_var.set('ステータス: 重複検出中...'))
        except Exception:
            pass
    total = len(compare_list)
    processed = 0
    results: list[tuple[str, str, str]] = []

    def worker(path: str):
        if cancel_flag.is_set():
            return None
        try:
            size = os.path.getsize(path)
        except Exception:
            return None

        # 同じフォルダ比較時用：このファイルの正規化パス
        path_normalized = normalize(path) if is_same_folder else ""

        name = os.path.basename(path)
        primary_candidates = list(main_lookup.get((name, size), []))
        for candidate in primary_candidates:
            if cancel_flag.is_set():
                return None
            if not os.path.exists(candidate):
                continue
            # 同じフォルダ比較時：同一ファイルパス同士の比較はスキップ
            if is_same_folder and normalize(candidate) == path_normalized:
                continue
            # 同じフォルダ比較時：辞書順で候補が現在のパスより大きい場合はスキップ（逆順での重複検出を防止）
            if is_same_folder and path_normalized and normalize(candidate) > path_normalized:
                continue
            return (candidate, path, 'name+size')

        size_candidates = main_size_index.get(size)
        if not size_candidates:
            return None

        compare_hash: Optional[str] = None
        for candidate in size_candidates:
            if cancel_flag.is_set():
                return None
            if not os.path.exists(candidate):
                continue
            if candidate in primary_candidates:
                continue
            # 同じフォルダ比較時：同一ファイルパス同士の比較はスキップ
            if is_same_folder and normalize(candidate) == path_normalized:
                continue
            # 同じフォルダ比較時：辞書順で候補が現在のパスより大きい場合はスキップ（逆順での重複検出を防止）
            if is_same_folder and path_normalized and normalize(candidate) > path_normalized:
                continue

            main_hash = get_main_hash(candidate)
            if not main_hash:
                continue

            if compare_hash is None:
                compare_hash = compute_hash(path)
            if not compare_hash:
                break

            if main_hash == compare_hash:
                return (candidate, path, 'size+hash')
        return None

    with ThreadPoolExecutor(max_workers=THREADS) as exe:
        futures = {exe.submit(worker, p): p for p in compare_list}
        for fut in as_completed(futures):
            if cancel_flag.is_set():
                break
            res = fut.result()
            processed += 1
            if progress is not None:
                value = (processed / max(1, total)) * 100
                tk_root.after(0, lambda val=value: _update_progress(progress, val))
            if res:
                results.append(res)

    # 同じフォルダ比較時：重複ペアの逆順を除去（念のための二重チェック）
    if is_same_folder and results:
        original_count = len(results)
        seen_pairs = set()
        unique_results = []
        for main_path, compare_path, reason in results:
            # 正規化パスのペアを作成（順序を統一）
            pair = tuple(sorted([normalize(main_path), normalize(compare_path)]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_results.append((main_path, compare_path, reason))
        if len(unique_results) < original_count and log_callback:
            removed = original_count - len(unique_results)
            log_callback(f'重複ペアの逆順を除去: {removed} 件（最終結果: {len(unique_results)} 件）')
        results = unique_results

    # 同じフォルダ比較の場合、重複ファイルの集計方法を変更
    if is_same_folder:
        # 重複ペアに含まれるすべてのファイルを収集
        all_duplicates = set()
        for main_path, compare_path, _ in results:
            all_duplicates.add(normalize(main_path))
            all_duplicates.add(normalize(compare_path))
        
        # 非重複ファイル = 全ファイル - 重複ファイル
        all_files_norm = {normalize(p) for p in main_files}
        non_duplicates_norm = all_files_norm - all_duplicates
        
        # 非重複ファイルのパスを取得
        compare_non_duplicates = [p for p in compare_list if normalize(p) in non_duplicates_norm]
        main_non_duplicates = []  # 同じフォルダなのでα側の非重複は0
        
        # バイトサイズ計算（重複ペア数ではなく、重複ファイル数分）
        total_bytes = 0
        counted_files = set()
        for main_path, compare_path, _ in results:
            for path in [main_path, compare_path]:
                norm_path = normalize(path)
                if norm_path not in counted_files:
                    counted_files.add(norm_path)
                    try:
                        if os.path.exists(path):
                            total_bytes += os.path.getsize(path)
                    except Exception:
                        continue
    else:
        # 異なるフォルダ比較の場合（従来の動作）
        duplicate_norms_compare = set()
        matched_main_norms = set()
        total_bytes = 0
        for main_path, compare_path, _ in results:
            duplicate_norms_compare.add(normalize(compare_path))
            matched_main_norms.add(normalize(main_path))
            try:
                if os.path.exists(main_path):
                    total_bytes += os.path.getsize(main_path)
            except Exception:
                continue

        compare_non_duplicates = [p for p in compare_list if normalize(p) not in duplicate_norms_compare]
        main_non_duplicates = [p for p in main_files if normalize(p) not in matched_main_norms]

    def apply_results():
        if compare_nondup_container is not None:
            compare_nondup_container.clear()
            compare_nondup_container.extend(compare_non_duplicates)
        if main_nondup_container is not None:
            main_nondup_container.clear()
            main_nondup_container.extend(main_non_duplicates)
        tree.delete(*tree.get_children())
        for r in results:
            # チェックボックス列を追加（デフォルトは未選択）
            tree.insert('', 'end', values=('☐', r[0], r[1], r[2]))
        count_label.config(
            text=(
                f'重複検出数: {len(results)} 件 / '
                f'非重複(α): {len(main_non_duplicates)} 件 / '
                f'非重複(β): {len(compare_non_duplicates)} 件'
            )
        )
        size_var.set(f'重複合計サイズ: {format_size(total_bytes)}')
        if open_compare_button is not None:
            state = 'normal' if compare_non_duplicates else 'disabled'
            open_compare_button.config(state=state)
        if open_main_button is not None:
            state = 'normal' if main_non_duplicates else 'disabled'
            open_main_button.config(state=state)
        if status_var is not None:
            status_var.set(f'ステータス: 重複検出完了 ({len(results)} 件)')
        if log_callback is not None:
            try:
                log_text = (
                    f'重複検出: 重複 {len(results)} 件 / '
                    f'非重複(α) {len(main_non_duplicates)} 件 / '
                    f'非重複(β) {len(compare_non_duplicates)} 件'
                )
                log_callback(log_text)
            except Exception:
                pass

    tk_root.after(0, apply_results)

    with open('duplicate_report.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['α','β','判定内容'])
        for r in results:
            w.writerow(r)


def _update_progress(progress_bar, value):
    try:
        progress_bar['value'] = value
        progress_bar.update_idletasks()
    except Exception:
        pass

# ---- リンク作成 ----
def run_mklink(link_path: str, target_path: str, mode: str) -> tuple[bool, str]:
    # mklinkはcmd.exeの内部コマンドなので、cmd /cで実行する
    # shell=Falseを使用してセキュリティを向上（OSコマンドインジェクション対策）
    if mode == 'hardlink':
        cmd = ['cmd', '/c', 'mklink', '/H', link_path, target_path]
    elif mode == 'junction':
        cmd = ['cmd', '/c', 'mklink', '/J', link_path, target_path]
    elif mode == 'symbolic_dir':
        cmd = ['cmd', '/c', 'mklink', '/D', link_path, target_path]
    else:
        cmd = ['cmd', '/c', 'mklink', link_path, target_path]
    try:
        res = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=CREATE_NO_WINDOW)
        ok = res.returncode == 0
        err = res.stderr.strip() or res.stdout.strip()
        return ok, err
    except Exception as e:
        return False, str(e)


def create_shortcut(link_path: Path, target_path: Path) -> tuple[bool, str]:
    ok, msg = _run_shortcut_batch(
        [
            (
                str(link_path),
                str(target_path),
            )
        ]
    )
    return ok, msg


def create_link(target_path: Path, link_path: Path, link_type: str, use_relative_path: bool = False) -> tuple[bool, str, Optional[Path]]:
    target = Path(target_path)
    link_location = Path(link_path)

    if not target.exists():
        return False, 'target_missing', None

    link_actual = link_location
    removal_candidates = [link_location]

    if link_type == 'ショートカット':
        suffix = link_actual.suffix
        if suffix.lower() != '.lnk':
            link_actual = link_actual.with_suffix(suffix + '.lnk' if suffix else '.lnk')
            if link_actual not in removal_candidates:
                removal_candidates.append(link_actual)

    for candidate in removal_candidates:
        if candidate.exists() or candidate.is_symlink():
            try:
                if candidate.is_dir() and not candidate.is_symlink():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                return False, f'failed_to_remove:{candidate}:{exc}', None

    link_actual.parent.mkdir(parents=True, exist_ok=True)

    # 相対パスを計算（ショートカット/シンボリックリンクのみ）
    target_for_link = target
    if use_relative_path and link_type in ['シンボリックリンク', 'ショートカット']:
        try:
            target_for_link = Path(os.path.relpath(target, link_actual.parent))
        except (ValueError, OSError):
            # 異なるドライブなど相対パス化できない場合は絶対パスのまま
            pass

    try:
        if link_type == 'ハードリンク':
            if target.is_dir():
                return False, 'hardlink_requires_file', None
            os.link(str(target), str(link_actual))
        elif link_type == 'ジャンクション':
            if not target.is_dir():
                return False, 'junction_requires_directory', None
            ok, msg = run_mklink(str(link_actual), str(target), mode='junction')
            if not ok:
                return False, msg, None
        elif link_type == 'シンボリックリンク':
            mode = 'symbolic_dir' if target.is_dir() else 'symbolic_file'
            ok, msg = run_mklink(str(link_actual), str(target_for_link), mode=mode)
            if not ok:
                return False, msg, None
        elif link_type == 'ショートカット':
            ok, msg = create_shortcut(link_actual, target_for_link)
            if not ok:
                return False, msg, None
        else:
            return False, f'unsupported_link_type:{link_type}', None
        return True, 'link_created', link_actual
    except Exception as exc:
        return False, str(exc), None

# ---- GUI ----
class App:
    def __init__(self, root):
        self.root = root
        root.title('Duplicate Link Tool')
        root.geometry('1000x850')
        root.minsize(950, 720)
        root.configure(bg=DARK_BG)
        try:
            icon_path = _resource_path(ASSET_ICON_NAME)
            if icon_path.exists():
                # iconbitmapでウィンドウアイコンを設定
                root.iconbitmap(default=str(icon_path))
                # wm_iconbitmapでタスクバーアイコンも明示的に設定
                root.wm_iconbitmap(str(icon_path))
        except Exception:
            pass

        style = ttk.Style()
        style.theme_use('default')
        style.configure('Treeview', background=DARK_BG, foreground=DARK_FG, fieldbackground=DARK_BG)
        style.configure('TButton', background=ACCENT, foreground=DARK_FG)
        style.configure('Horizontal.TProgressbar', troughcolor='#333333', background=ACCENT)
        style.map('Treeview', background=[('selected', '#2f4f6f')], foreground=[('selected', '#ffffff')])

        self.main_dir = tk.StringVar()
        self.compare_dir = tk.StringVar()
        self.realize_source = tk.StringVar()
        self.link_type = tk.StringVar(value='ショートカット')  # デフォルトをショートカットに変更
        self.link_name_mode = tk.StringVar(value='keep_main')  # デフォルトをα側に変更
        self.link_path_mode = tk.StringVar(value='relative')  # デフォルトを相対パスに変更
        self.keep_side = tk.StringVar(value='main')  # 'main' or 'compare' - どちらを残すか
        self.use_same_as_main = tk.BooleanVar(value=False)
        self.cancel_flag = threading.Event()
        self.compare_non_duplicates = []
        self.main_non_duplicates = []
        self.main_link_summary_var = tk.StringVar(value='リンク状況: 未選択')
        self.compare_link_summary_var = tk.StringVar(value='リンク状況: 未選択')
        self.realize_link_summary_var = tk.StringVar(value='リンク状況: 未選択')

        top = tk.Frame(root, bg=DARK_BG)
        top.pack(padx=10, pady=10, fill='x')
        top.columnconfigure(0, weight=1)

        tk.Label(top, text='1) αディレクトリ', bg=DARK_BG, fg=DARK_FG).grid(row=0, column=0, sticky='w')
        self.main_entry = HistoryEntry(top, textvariable=self.main_dir, width=80, bg='#2b2b2b', fg=DARK_FG)
        self.main_entry.grid(row=1, column=0, sticky='we')
        tk.Button(top, text='選択', command=self.select_main).grid(row=1, column=1, padx=5)
        tk.Label(top, textvariable=self.main_link_summary_var, bg=DARK_BG, fg=DARK_FG).grid(row=2, column=0, columnspan=2, sticky='w', pady=(2,0))

        tk.Label(top, text='2) βディレクトリ', bg=DARK_BG, fg=DARK_FG).grid(row=3, column=0, sticky='w', pady=(8,0))
        compare_row_frame = tk.Frame(top, bg=DARK_BG)
        compare_row_frame.grid(row=4, column=0, columnspan=2, sticky='we')
        compare_row_frame.columnconfigure(0, weight=1)
        
        self.compare_entry = HistoryEntry(compare_row_frame, textvariable=self.compare_dir, width=80, bg='#2b2b2b', fg=DARK_FG)
        self.compare_entry.grid(row=0, column=0, sticky='we')
        tk.Button(compare_row_frame, text='選択', command=self.select_compare).grid(row=0, column=1, padx=5)
        self.same_as_main_cb = tk.Checkbutton(
            compare_row_frame,
            text='αと同じ',
            variable=self.use_same_as_main,
            command=self.on_same_as_main_toggle,
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
            activebackground=DARK_BG,
            activeforeground=DARK_FG
        )
        self.same_as_main_cb.grid(row=0, column=2, padx=5)
        
        tk.Label(top, textvariable=self.compare_link_summary_var, bg=DARK_BG, fg=DARK_FG).grid(row=5, column=0, columnspan=2, sticky='w', pady=(2,0))

        tk.Label(top, text='実体化して圧縮するディレクトリ', bg=DARK_BG, fg=DARK_FG).grid(row=6, column=0, sticky='w', pady=(8,0))
        self.realize_entry = HistoryEntry(top, textvariable=self.realize_source, width=80, bg='#2b2b2b', fg=DARK_FG)
        self.realize_entry.grid(row=7, column=0, sticky='we')
        tk.Button(top, text='選択', command=self.select_realize_source).grid(row=7, column=1, padx=5)
        tk.Label(top, textvariable=self.realize_link_summary_var, bg=DARK_BG, fg=DARK_FG).grid(row=8, column=0, columnspan=2, sticky='w', pady=(2,0))

        tk.Label(top, text='リンク種別', bg=DARK_BG, fg=DARK_FG).grid(row=9, column=0, sticky='w', pady=(8,0))
        types_frame = tk.Frame(top, bg=DARK_BG)
        types_frame.grid(row=10, column=0, columnspan=2, sticky='w')
        for label_text, value in [
            ('ショートカット', 'ショートカット'),
            ('シンボリックリンク', 'シンボリックリンク'),
            ('ハードリンク', 'ハードリンク'),
            ('ジャンクション (フォルダのみ)', 'ジャンクション'),
        ]:
            tk.Radiobutton(
                types_frame,
                text=label_text,
                variable=self.link_type,
                value=value,
                bg=DARK_BG,
                fg=DARK_FG,
                selectcolor=DARK_BG,
            ).pack(side='left', padx=4)

        tk.Label(top, text='リンクパスの形式 (ショートカット/シンボリックリンクのみ)', bg=DARK_BG, fg=DARK_FG).grid(row=11, column=0, sticky='w', pady=(10,0))
        path_mode_frame = tk.Frame(top, bg=DARK_BG)
        path_mode_frame.grid(row=12, column=0, columnspan=2, sticky='w')
        tk.Radiobutton(
            path_mode_frame,
            text='相対パス',
            variable=self.link_path_mode,
            value='relative',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)
        tk.Radiobutton(
            path_mode_frame,
            text='絶対パス',
            variable=self.link_path_mode,
            value='absolute',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)

        tk.Label(top, text='リンク作成後に残すファイル名', bg=DARK_BG, fg=DARK_FG).grid(row=13, column=0, sticky='w', pady=(10,0))
        name_frame = tk.Frame(top, bg=DARK_BG)
        name_frame.grid(row=14, column=0, columnspan=2, sticky='w')
        tk.Radiobutton(
            name_frame,
            text='α側の名前を残す',
            variable=self.link_name_mode,
            value='keep_main',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)
        tk.Radiobutton(
            name_frame,
            text='β側の名前を残す',
            variable=self.link_name_mode,
            value='keep_compare',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)

        tk.Label(top, text='リンク化実行時にどちらを残すか', bg=DARK_BG, fg=DARK_FG).grid(row=15, column=0, sticky='w', pady=(10,0))
        keep_side_frame = tk.Frame(top, bg=DARK_BG)
        keep_side_frame.grid(row=16, column=0, columnspan=2, sticky='w')
        tk.Radiobutton(
            keep_side_frame,
            text='αを残す（β側をリンク化）',
            variable=self.keep_side,
            value='main',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)
        tk.Radiobutton(
            keep_side_frame,
            text='βを残す（α側をリンク化）',
            variable=self.keep_side,
            value='compare',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)

        btn_frame = tk.Frame(root, bg=DARK_BG)
        btn_frame.pack(pady=6, fill='x', padx=10)
        
        # ボタンを中央に配置するための内部フレーム
        btn_inner = tk.Frame(btn_frame, bg=DARK_BG)
        btn_inner.pack()
        
        self.detect_btn = tk.Button(btn_inner, text='重複を検出', command=self.detect)
        self.detect_btn.pack(side='left', padx=6)
        self.link_btn = tk.Button(btn_inner, text='リンク化を実行', command=self.run_linking)
        self.link_btn.pack(side='left', padx=6)
        self.cancel_btn = tk.Button(btn_inner, text='キャンセル', command=self.cancel_process)
        self.cancel_btn.pack(side='left', padx=6)
        self.open_main_nondup_btn = tk.Button(
            btn_inner,
            text='非重複(α)を開く',
            command=lambda: self.open_non_duplicates('main'),
            state='disabled',
        )
        self.open_main_nondup_btn.pack(side='left', padx=6)
        self.open_compare_nondup_btn = tk.Button(
            btn_inner,
            text='非重複(β)を開く',
            command=lambda: self.open_non_duplicates('compare'),
            state='disabled',
        )
        self.open_compare_nondup_btn.pack(side='left', padx=6)
        self.materialize_btn = tk.Button(btn_inner, text='リンクを実体化', command=self.materialize_links)
        self.materialize_btn.pack(side='left', padx=6)
        self.realize_btn = tk.Button(btn_inner, text='実体化して圧縮', command=self.realize_and_zip)
        self.realize_btn.pack(side='left', padx=6)
        tk.Button(btn_inner, text='CSV を開く', command=self.open_csv).pack(side='left', padx=6)

        self.progress = ttk.Progressbar(root, mode='determinate', style='Horizontal.TProgressbar')
        self.progress.pack(pady=8, fill='x', padx=10)
        
        self.count_label = tk.Label(
            root,
            text='重複検出数: 0 件 / 非重複(α): 0 件 / 非重複(β): 0 件',
            bg=DARK_BG,
            fg=DARK_FG,
        )
        self.count_label.pack(padx=10)
        self.total_size_var = tk.StringVar(value='重複合計サイズ: 0 B')
        self.total_size_label = tk.Label(root, textvariable=self.total_size_var, bg=DARK_BG, fg=DARK_FG)
        self.total_size_label.pack(padx=10)
        self.zip_status_var = tk.StringVar(value='ステータス: 待機中')
        self.zip_status_label = tk.Label(root, textvariable=self.zip_status_var, bg=DARK_BG, fg=DARK_FG)
        self.zip_status_label.pack(padx=10)

        tree_container = tk.Frame(root, bg=DARK_BG)
        tree_container.pack(fill='both', expand=True, padx=10, pady=(6, 3))

        tree_frame = tk.Frame(tree_container, bg=DARK_BG)
        tree_frame.pack(fill='both', expand=True, padx=0, pady=0)
        self.tree = ttk.Treeview(tree_frame, columns=('select', 'main', 'dup', 'reason'), show='headings', height=13, selectmode='browse')
        self.tree.heading('select', text='☐', command=lambda: self.toggle_all_selection())
        self.tree.heading('main', text='α')
        self.tree.heading('dup', text='β')
        self.tree.heading('reason', text='判定内容')
        self.tree.column('select', width=40, anchor='center', stretch=False)
        self.tree.column('main', width=360)
        self.tree.column('dup', width=360)
        self.tree.column('reason', width=200)
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        tree_scroll_y.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True, side='left')
        
        # 選択状態を管理する辞書（item_id → bool）
        self.tree_selection_state = {}
        # 最後に選択されたアイテム（Shift選択用）
        self.last_selected_item = None
        # ヘッダーチェックボックスの不要な再トグルを防ぐフラグ
        self._suppress_header_toggle = False

        log_container = tk.Frame(root, bg=DARK_BG, height=180)
        log_container.pack(fill='both', expand=False, padx=10, pady=(0, 10))
        log_container.pack_propagate(False)

        tk.Label(log_container, text='ログ', bg=DARK_BG, fg=DARK_FG, anchor='w').pack(fill='x', padx=0)
        log_frame = tk.Frame(log_container, bg=DARK_BG)
        log_frame.pack(fill='both', expand=True, padx=0)
        self.log = tk.Text(log_frame, bg='#2b2b2b', fg=DARK_FG, wrap='word')
        log_scroll_y = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll_y.set)
        log_scroll_y.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True, side='left')

        # Treeviewのイベントバインド
        self.tree.bind('<ButtonRelease-1>', self.on_tree_click)
        self.tree.bind('<Button-1>', self.on_tree_button_down)
        self._adjust_initial_geometry()
        self.root.after(200, self._ensure_front)
    
    def toggle_all_selection(self):
        """全選択/全解除を切り替え"""
        if self._suppress_header_toggle:
            self._suppress_header_toggle = False
            return

        all_items = self.tree.get_children()
        if not all_items:
            return
        
        # 現在の選択状態を確認
        selected_count = sum(1 for item in all_items if self.tree_selection_state.get(item, False))
        
        # 半分以上選択されていれば全解除、そうでなければ全選択
        new_state = selected_count < len(all_items) / 2
        
        for item in all_items:
            self.tree_selection_state[item] = new_state
            values = list(self.tree.item(item, 'values'))
            values[0] = '☑' if new_state else '☐'
            self.tree.item(item, values=values)
        
        # ヘッダーのチェックボックスも更新
        header_text = '☑' if new_state else '☐'
        self.tree.heading('select', text=header_text)
        
        # 最後の選択をリセット（範囲選択の開始点をクリア）
        self.last_selected_item = None
    
    def on_tree_button_down(self, event):
        """マウスボタン押下時の処理（ヘッダークリックの判定用）"""
        region = self.tree.identify_region(event.x, event.y)
        if region == 'heading':
            # ヘッダー領域のクリックは処理を続行（カラムリサイズやソート用）
            return
        # それ以外は通常のクリックとして処理
    
    def on_tree_click(self, event):
        """Treeviewクリック時の処理"""
        region = self.tree.identify_region(event.x, event.y)
        
        # ヘッダー領域のクリック
        if region == 'heading':
            column = self.tree.identify_column(event.x)
            # select列のヘッダークリックは全選択/全解除
            if column == '#1':
                self.toggle_all_selection()
            # その他のヘッダーは無視（下のアイテムへの伝播を防止）
            return 'break'
        
        # セパレータ（カラムリサイズ）のクリックも無視
        if region == 'separator':
            return 'break'
        
        # アイテム領域のクリック
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        
        column = self.tree.identify_column(event.x)
        
        # チェックボックス列のクリック
        if column == '#1':  # select列
            shift_pressed = (event.state & 0x0001) != 0  # Shiftキーの状態

            # last_selected_itemが有効かチェック（削除されていないか、存在するか）
            last_item_exists = (
                self.last_selected_item is not None
                and self.last_selected_item in self.tree.get_children()
            )

            # ヘッダーの全選択トグルを一時的に抑制
            self._suppress_header_toggle = True
            self.root.after(0, lambda: setattr(self, '_suppress_header_toggle', False))

            if shift_pressed and last_item_exists:
                # Shift+クリック：範囲選択/解除
                # クリックしたアイテムの現在の状態に基づいて範囲全体を切り替え
                current_state = self.tree_selection_state.get(item_id, False)
                new_state = not current_state
                self._range_toggle(self.last_selected_item, item_id, new_state)
                self.last_selected_item = item_id
            else:
                # 通常クリック：単一トグル
                current_state = self.tree_selection_state.get(item_id, False)
                new_state = not current_state
                self.tree_selection_state[item_id] = new_state

                values = list(self.tree.item(item_id, 'values'))
                values[0] = '☑' if new_state else '☐'
                self.tree.item(item_id, values=values)

                self.last_selected_item = item_id
            return 'break'
        else:
            # その他の列：ファイルを開く
            self.tree.selection_set(item_id)
            self.tree.focus(item_id)
            self.root.after(50, lambda: self._open_pair_from_item(item_id))
    
    def _range_toggle(self, start_item, end_item, new_state):
        """範囲選択/解除を実行"""
        all_items = self.tree.get_children()
        try:
            start_idx = all_items.index(start_item)
            end_idx = all_items.index(end_item)
        except ValueError:
            return
        
        # 範囲を決定
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        # 範囲内の全アイテムを指定された状態に設定
        for i in range(start_idx, end_idx + 1):
            item = all_items[i]
            self.tree_selection_state[item] = new_state
            values = list(self.tree.item(item, 'values'))
            values[0] = '☑' if new_state else '☐'
            self.tree.item(item, values=values)
        
        # 範囲選択後はlast_selected_itemを更新しない
        # これにより次のクリックが新しい起点として扱われる
    
    def log_message(self, msg):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {msg}\n'

        def append_to_widget():
            try:
                self.log.insert('end', line)
                self.log.see('end')
            except Exception:
                pass

        if threading.current_thread() is threading.main_thread():
            append_to_widget()
        else:
            self.root.after(0, append_to_widget)

        try:
            with open(f'link_tool_log_{datetime.datetime.now().strftime("%Y%m%d")}.txt', 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass

    def _ensure_front(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.attributes('-topmost', True)
            self.root.after(800, lambda: self.root.attributes('-topmost', False))
        except Exception:
            pass

    def _adjust_initial_geometry(self) -> None:
        try:
            # レイアウトを完全に更新するため複数回実行
            self.root.update_idletasks()
            self.root.update()
            self.root.update_idletasks()
            
            # 要求された幅と高さを取得
            required_w = max(1000, self.root.winfo_reqwidth())
            required_h = max(850, self.root.winfo_reqheight())
            
            # 幅を少し広めに設定してガタツキを防ぐ
            adjusted_w = required_w + 10
            self.root.geometry(f'{adjusted_w}x{required_h}')
            
            # 最終的なレイアウト更新
            self.root.update_idletasks()
        except Exception:
            pass
    
    def get_selected_items(self):
        """チェックボックスで選択されたアイテムのリストを取得"""
        selected = []
        for item_id in self.tree.get_children():
            if self.tree_selection_state.get(item_id, False):
                values = self.tree.item(item_id, 'values')
                if len(values) >= 4:  # select, main, dup, reason
                    selected.append({
                        'item_id': item_id,
                        'main': values[1],
                        'duplicate': values[2],
                        'reason': values[3]
                    })
        return selected

    def _format_link_summary(self, counts: dict[str, int]) -> str:
        parts = []
        if counts.get('shortcut'):
            parts.append(f"ショートカット {counts['shortcut']}件")
        if counts.get('symlink'):
            parts.append(f"シンボリックリンク {counts['symlink']}件")
        if counts.get('hardlink'):
            parts.append(f"ハードリンク {counts['hardlink']}件")
        if counts.get('junction'):
            parts.append(f"ジャンクション {counts['junction']}件")
        if not parts:
            return 'リンク状況: リンクなし'
        return 'リンク状況: ' + ', '.join(parts)

    def _link_summary_worker(self, root_path: Path, var: tk.StringVar) -> None:
        counts = {'symlink': 0, 'junction': 0, 'hardlink': 0, 'shortcut': 0}
        try:
            for dirpath, dirnames, filenames in os.walk(root_path):
                for name in dirnames + filenames:
                    path = Path(dirpath) / name
                    try:
                        st = os.lstat(path)
                    except Exception:
                        continue
                    
                    # ショートカットの判定
                    if path.suffix.lower() == '.lnk':
                        counts['shortcut'] += 1
                        continue
                    
                    # シンボリックリンクの判定（ファイルとディレクトリ両方）
                    if stat.S_ISLNK(st.st_mode):
                        # ジャンクション（ディレクトリのリパースポイント）とシンボリックリンクを区別
                        if path.is_dir() and (getattr(st, 'st_file_attributes', 0) & FILE_ATTRIBUTE_REPARSE_POINT):
                            counts['junction'] += 1
                        else:
                            counts['symlink'] += 1
                        continue
                    
                    # ハードリンクの判定（ファイルのみ、リンク数が2以上）
                    if not path.is_dir() and getattr(st, 'st_nlink', 1) > 1:
                        counts['hardlink'] += 1
                        continue
                    
                    # ジャンクション（stat.S_ISLNKでキャッチされなかった場合の追加チェック）
                    if (getattr(st, 'st_file_attributes', 0) & FILE_ATTRIBUTE_REPARSE_POINT):
                        counts['junction'] += 1
        except Exception as exc:
            summary = f'リンク状況: 解析失敗 ({exc})'
        else:
            summary = self._format_link_summary(counts)
        self.root.after(0, lambda: var.set(summary))

    def _schedule_link_summary(self, directory: str, var: tk.StringVar) -> None:
        if not directory:
            var.set('リンク状況: 未選択')
            return
        path = Path(directory)
        if not path.exists():
            var.set('リンク状況: パスが存在しません')
            return
        var.set('リンク状況: 集計中...')
        threading.Thread(target=self._link_summary_worker, args=(path, var), daemon=True).start()

    def select_main(self):
        d = filedialog.askdirectory()
        if d:
            self.main_dir.set(d)
            self.log_message(f'αディレクトリ: {d}')
            self._schedule_link_summary(d, self.main_link_summary_var)
            # 「αと同じ」がチェックされている場合はβ側も更新
            if self.use_same_as_main.get():
                self.compare_dir.set(d)
                self.log_message(f'βディレクトリをαと同じに設定: {d}')
                self._schedule_link_summary(d, self.compare_link_summary_var)

    def select_compare(self):
        d = filedialog.askdirectory()
        if d:
            self.compare_dir.set(d)
            self.log_message(f'βディレクトリ: {d}')
            self._schedule_link_summary(d, self.compare_link_summary_var)
    
    def on_same_as_main_toggle(self):
        """「αと同じ」チェックボックスの状態変更時の処理"""
        if self.use_same_as_main.get():
            # チェックされた：αと同じパスを設定
            main_path = self.main_dir.get()
            if main_path:
                self.compare_dir.set(main_path)
                self.log_message(f'βディレクトリをαと同じに設定: {main_path}')
                self._schedule_link_summary(main_path, self.compare_link_summary_var)
            self.compare_entry.config(state='disabled', disabledbackground='#555555', disabledforeground='#999999')
        else:
            # チェック解除：入力可能に戻す
            self.compare_entry.config(state='normal', bg='#2b2b2b', fg=DARK_FG)

    def select_realize_source(self):
        d = filedialog.askdirectory()
        if d:
            self.realize_source.set(d)
            self.log_message(f'実体化対象ディレクトリ: {d}')
            self._schedule_link_summary(d, self.realize_link_summary_var)

    def detect(self):
        if not self.main_dir.get() or not self.compare_dir.get():
            messagebox.showwarning('警告', '両方のディレクトリを選択してください')
            return
        self.tree.delete(*self.tree.get_children())
        self.tree_selection_state.clear()  # 選択状態をクリア
        self.last_selected_item = None  # 最後の選択をリセット
        self.log.delete('1.0', 'end')
        self.progress['value'] = 0
        self.cancel_flag.clear()
        self.compare_non_duplicates.clear()
        self.main_non_duplicates.clear()
        self.open_compare_nondup_btn.config(state='disabled')
        self.open_main_nondup_btn.config(state='disabled')
        self.count_label.config(
            text='重複検出数: 集計中... / 非重複(α): 集計中... / 非重複(β): 集計中...'
        )
        self.zip_status_var.set('ステータス: 重複検出を準備しています...')
        self.total_size_var.set('重複合計サイズ: 計算中...')
        
        # Treeviewのヘッダーをフォルダ名で更新
        main_folder_name = os.path.basename(self.main_dir.get().rstrip(os.sep))
        compare_folder_name = os.path.basename(self.compare_dir.get().rstrip(os.sep))
        self.tree.heading('main', text=main_folder_name or 'α')
        self.tree.heading('dup', text=compare_folder_name or 'β')
        
        threading.Thread(
            target=find_duplicates,
            args=(
                self.main_dir.get(),
                self.compare_dir.get(),
                self.progress,
                self.tree,
                self.count_label,
                self.total_size_var,
                self.zip_status_var,
                self.compare_non_duplicates,
                self.main_non_duplicates,
                self.open_compare_nondup_btn,
                self.open_main_nondup_btn,
                self.cancel_flag,
                self.root,
                self.log_message,
            ),
            daemon=True,
        ).start()
        self.log_message('重複検出を開始しました')

    def _open_pair_from_item(self, item_id):
        values = self.tree.item(item_id, 'values')
        if len(values) < 3:  # select, main, dup, reason
            return
        # values[0]はチェックボックス、values[1]がα、values[2]がβ
        main_path, duplicate_path = values[1], values[2]
        opened_any = False

        for label, path in (('α', main_path), ('β', duplicate_path)):
            if not path:
                continue
            if self._open_in_explorer(path):
                opened_any = True
            else:
                self.log_message(f'[WARN] {label} パスを開けませんでした: {path}')

        if not opened_any:
            messagebox.showwarning('警告', '対象ファイルをエクスプローラーで開けませんでした。')

    def _open_in_explorer(self, path: str) -> bool:
        try:
            p = Path(path)
            if not p.exists():
                return False
            subprocess.Popen(['explorer', '/select,', str(p.resolve())])
            return True
        except Exception as e:
            self.log_message(f'[ERROR] Explorer 起動エラー: {e}')
            return False

    def open_non_duplicates(self, side: str):
        targets = self.main_non_duplicates if side == 'main' else self.compare_non_duplicates
        label = 'α' if side == 'main' else 'β'

        if not targets:
            messagebox.showinfo('情報', f'{label}側で重複として検出されなかったファイルはありません。')
            return

        opened = 0
        for path in targets:
            if self._open_in_explorer(path):
                opened += 1
            else:
                self.log_message(f'[WARN] 非重複ファイルを開けませんでした ({label}): {path}')

        if opened == 0:
            messagebox.showwarning('警告', f'非重複ファイルをエクスプローラーで開けませんでした ({label}側)。')
        else:
            self.log_message(f'{label}側の非重複ファイル {opened} 件をエクスプローラーで開きました')

    def _set_action_buttons(self, busy: bool):
        state = 'disabled' if busy else 'normal'
        self.detect_btn.config(state=state)
        self.link_btn.config(state=state)
        self.cancel_btn.config(state=state)
        self.realize_btn.config(state=state)
        self.materialize_btn.config(state=state)
        if busy:
            self.open_compare_nondup_btn.config(state='disabled')
            self.open_main_nondup_btn.config(state='disabled')
        else:
            self.open_compare_nondup_btn.config(
                state='normal' if self.compare_non_duplicates else 'disabled'
            )
            self.open_main_nondup_btn.config(
                state='normal' if self.main_non_duplicates else 'disabled'
            )

    def _set_realize_progress(self, value: float, status_text: str):
        self.progress['value'] = value
        self.zip_status_var.set(status_text)

    def realize_and_zip(self):
        source_value = self.realize_source.get()
        if not source_value:
            messagebox.showwarning('警告', '実体化して圧縮するディレクトリを選択してください。')
            return

        source_path = Path(source_value)
        if not source_path.exists():
            messagebox.showerror('エラー', f'実体化ディレクトリが存在しません: {source_path}')
            return

        default_name = f'{source_path.name}_realized.zip' if source_path.name else 'realized.zip'
        zip_dest = filedialog.asksaveasfilename(
            title='ZIP の保存先',
            defaultextension='.zip',
            initialfile=default_name,
            initialdir=source_path.parent,
            filetypes=[('ZIP ファイル', '*.zip')]
        )
        if not zip_dest:
            return

        temp_dir = BASE_DIR / 'RealizedCopy'
        self._set_action_buttons(True)
        self.log_message(f'実体化圧縮を開始: {source_path} -> {zip_dest}')
        self._set_realize_progress(0, 'ステータス: 実体化コピーを準備しています...')

        threading.Thread(
            target=self._realize_and_zip_worker,
            args=(source_path, Path(zip_dest), temp_dir),
            daemon=True,
        ).start()

    def _realize_and_zip_worker(self, source_path: Path, zip_path: Path, temp_dir: Path):
        warnings = []

        def progress_cb(copied: int, total: int):
            effective_total = total if total else 1
            percent = (copied / effective_total) * 80
            status = f'ステータス: 実体化コピー中 ({copied}/{total})' if total else 'ステータス: 実体化コピー中 (0 件)'
            self.root.after(0, lambda: self._set_realize_progress(percent, status))

        try:
            cleanup_directory(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            copied, total, warnings = realize_directory(source_path, temp_dir, progress_cb)
            self.root.after(0, lambda: self._set_realize_progress(85, 'ステータス: 圧縮中...'))

            zip_path.parent.mkdir(parents=True, exist_ok=True)
            if zip_path.exists():
                zip_path.unlink()
            zip_directory(temp_dir, zip_path)

            self.root.after(0, lambda: self._set_realize_progress(95, 'ステータス: 後片付け中...'))
            cleanup_directory(temp_dir)
            self.root.after(0, lambda: self._on_realize_success(zip_path, copied, total, warnings))

        except Exception as e:
            cleanup_directory(temp_dir)
            err_text = ''.join(traceback.format_exception_only(type(e), e)).strip()
            detail = traceback.format_exc()
            self.root.after(0, lambda: self._on_realize_failure(err_text, detail))

    def _on_realize_success(self, zip_path: Path, copied: int, total: int, warnings):
        self._set_realize_progress(100, f'ステータス: 完了 ({zip_path.name})')
        self.log_message(f'実体化圧縮が完了しました: {zip_path} ({copied} ファイル)')
        for warn in warnings:
            self.log_message(f'[WARN] {warn}')
        self._set_action_buttons(False)
        messagebox.showinfo('完了', f'ZIP を作成しました:\n{zip_path}')

    def _on_realize_failure(self, short_error: str, detail: str):
        self._set_realize_progress(0, 'ステータス: エラーが発生しました')
        self.log_message(f'[ERROR] 実体化圧縮に失敗しました: {short_error}')
        for line in detail.strip().splitlines():
            self.log_message(line)
        self._set_action_buttons(False)
        messagebox.showerror('エラー', f'実体化圧縮に失敗しました:\n{short_error}')

    def _describe_link_error(self, raw: str) -> str:
        mapping = {
            'target_missing': '元ファイルが見つかりません',
            'hardlink_requires_file': 'ハードリンクはファイルにのみ適用できます',
            'junction_requires_directory': 'ジャンクションはフォルダにのみ適用できます',
            'powershell_not_found': 'PowerShell が見つからないためショートカットを作成できません',
            'unsupported_link_type': '未対応のリンク種別です',
        }
        if raw in mapping:
            return mapping[raw]
        if raw.startswith('failed_to_remove:'):
            parts = raw.split(':', 2)
            if len(parts) == 3:
                return f'既存パスを削除できませんでした: {parts[1]} ({parts[2]})'
        if raw.startswith('unsupported_link_type:'):
            _, _, detail = raw.partition(':')
            return f'未対応のリンク種別です: {detail}'
        return raw

    def _classify_link_path(self, path: Path) -> str:
        try:
            if path.suffix.lower() == '.lnk' and path.exists():
                return 'ショートカット'
        except Exception:
            pass

        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return 'missing'
        except Exception as exc:
            return f'unknown:{exc}'

        if stat.S_ISLNK(st.st_mode):
            return 'シンボリックリンク'

        if getattr(st, 'st_nlink', 1) > 1 and not path.is_dir():
            return 'ハードリンク'

        if getattr(st, 'st_file_attributes', 0) & FILE_ATTRIBUTE_REPARSE_POINT:
            return 'ジャンクション'

        return '実体'

    def _unique_path(self, base_dir: Path, prefix: str, suffix: str = '') -> Path:
        for _ in range(1000):
            candidate = base_dir / f'{prefix}_{uuid.uuid4().hex}{suffix}'
            if not candidate.exists():
                return candidate
        raise RuntimeError('一時パスを生成できませんでした')

    def _materialize_shortcut(self, shortcut: Path) -> tuple[bool, bool, Optional[Path], Optional[str]]:
        ok, target, message = resolve_shortcut(shortcut)
        if not ok or target is None:
            return True, False, None, message or 'ショートカット解決に失敗しました'
        target_path = Path(target)
        if not target_path.exists():
            return True, False, None, f'リンク先が存在しません: {target_path}'

        dest_path = shortcut.with_suffix('')
        if dest_path.exists():
            return True, False, None, f'同名の実体が既に存在します: {dest_path}'

        try:
            if target_path.is_dir():
                shutil.copytree(target_path, dest_path)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_path, dest_path)
            try:
                shortcut.unlink()
            except FileNotFoundError:
                pass
            return True, True, dest_path, None
        except Exception as exc:
            if dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path, ignore_errors=True)
                else:
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
            return True, False, None, str(exc)

    def _materialize_symlink(self, link: Path) -> tuple[bool, bool, Optional[Path], Optional[str]]:
        try:
            target = link.resolve(strict=False)
        except Exception as exc:
            return True, False, None, f'リンク解決に失敗しました: {exc}'

        if not target.exists():
            return True, False, None, f'リンク先が存在しません: {target}'

        if target.is_dir():
            temp_path = self._unique_path(link.parent, f'{link.name}_materialize_dir')
            try:
                shutil.copytree(target, temp_path)
                link.unlink()
                temp_path.rename(link)
                return True, True, link, None
            except Exception as exc:
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)
                return True, False, None, str(exc)
        else:
            temp_path = self._unique_path(link.parent, f'{link.stem}_materialize_file', suffix='.tmp')
            try:
                shutil.copy2(target, temp_path)
                link.unlink()
                temp_path.rename(link)
                return True, True, link, None
            except Exception as exc:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return True, False, None, str(exc)

    def _materialize_hardlink(self, path: Path) -> tuple[bool, bool, Optional[Path], Optional[str]]:
        temp_path = self._unique_path(path.parent, f'{path.stem}_materialize_hardlink', suffix=path.suffix or '.tmp')
        try:
            shutil.copy2(path, temp_path)
            os.replace(str(temp_path), str(path))
            return True, True, path, None
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return True, False, None, str(exc)

    def _materialize_entry(self, entry: Path) -> tuple[bool, bool, Optional[Path], Optional[str]]:
        try:
            if entry.suffix.lower() == '.lnk' and entry.is_file():
                return self._materialize_shortcut(entry)
            stats = os.lstat(entry)
        except FileNotFoundError:
            return True, False, None, '対象が見つかりません'
        except Exception as exc:
            return True, False, None, str(exc)

        try:
            if entry.is_symlink():
                return self._materialize_symlink(entry)
            if not entry.is_dir() and stats.st_nlink > 1:
                return self._materialize_hardlink(entry)
        except Exception as exc:
            return True, False, None, str(exc)

        return False, False, None, None

    def materialize_links(self):
        directory = filedialog.askdirectory(title='リンクを実体化するディレクトリを選択')
        if not directory:
            return
        target_path = Path(directory)
        if not target_path.exists():
            messagebox.showerror('エラー', f'ディレクトリが存在しません: {target_path}')
            return

        self._set_action_buttons(True)
        self.log_message(f'リンク実体化を開始: {target_path}')
        threading.Thread(target=self._materialize_links_worker, args=(target_path,), daemon=True).start()

    def _materialize_links_worker(self, root_path: Path):
        success = fail = 0
        failures: list[str] = []
        skipped = 0  # スキップされたファイル数

        try:
            for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
                base = Path(dirpath)
                for name in filenames:
                    handled, ok, out_path, message = self._materialize_entry(base / name)
                    if not handled:
                        skipped += 1
                        continue
                    display = out_path if out_path is not None else base / name
                    if ok:
                        success += 1
                        self.log_message(f'[OK] 実体化: {display}')
                    else:
                        fail += 1
                        detail = message or '理由不明'
                        failures.append(f'{display}: {detail}')
                        self.log_message(f'[FAIL] 実体化: {display} ({detail})')

                for name in dirnames:
                    handled, ok, out_path, message = self._materialize_entry(base / name)
                    if not handled:
                        skipped += 1
                        continue
                    display = out_path if out_path is not None else base / name
                    if ok:
                        success += 1
                        self.log_message(f'[OK] 実体化: {display}')
                    else:
                        fail += 1
                        detail = message or '理由不明'
                        failures.append(f'{display}: {detail}')
                        self.log_message(f'[FAIL] 実体化: {display} ({detail})')
        except Exception as exc:
            fail += 1
            failures.append(str(exc))
            self.log_message(f'[ERROR] 実体化処理中に例外が発生しました: {exc}')

        self.log_message(f'実体化処理統計: 処理対象={success + fail}件, スキップ={skipped}件（通常ファイル）')
        self.root.after(0, lambda: self._on_materialize_complete(root_path, success, fail, failures))

    def _on_materialize_complete(self, root_path: Path, success: int, fail: int, failures: list[str]):
        self._set_action_buttons(False)
        summary = f'リンク実体化が完了しました: 成功 {success} 件, 失敗 {fail} 件'
        self.log_message(summary)
        for item in failures:
            self.log_message(f'[WARN] 実体化に失敗: {item}')

        for directory, var in [
            (self.main_dir.get(), self.main_link_summary_var),
            (self.compare_dir.get(), self.compare_link_summary_var),
            (self.realize_source.get(), self.realize_link_summary_var),
        ]:
            if directory:
                self._schedule_link_summary(directory, var)

        messagebox.showinfo('完了', f'リンクの実体化が完了しました。\n成功: {success} 件\n失敗: {fail} 件')

    def _remove_existing_path(self, path: Path) -> tuple[bool, str]:
        try:
            if not path.exists() and not path.is_symlink():
                return True, ''
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True, ''
        except FileNotFoundError:
            return True, ''
        except Exception as exc:
            return False, str(exc)

    def _execute_shortcut_batch(self, operations: list[dict[str, Path]]) -> tuple[bool, str, str]:
        entries = []
        for entry in operations:
            link_path = entry['link']
            target_path = entry['target']
            entries.append((str(link_path), str(target_path)))
        script = _build_shortcut_batch_script(entries)
        ok, msg = _run_shortcut_batch(entries, script=script)
        return ok, msg, script

    def _linking_shortcut_batch(self, name_mode: str, use_relative_path: bool = False, keep_side: str = 'main') -> None:
        success = fail = 0
        created_counts: Counter[str] = Counter()
        cancelled = False
        operations: list[dict[str, Path]] = []
        contexts: list[tuple[Path, Path, Path, Path]] = []  # (link_actual, target_for_shortcut, original_to_delete, target_path)

        for item in self.tree.get_children():
            if self.cancel_flag.is_set():
                cancelled = True
                self.log_message('ユーザーがキャンセルしました')
                break

            # チェックボックスで選択されていないものはスキップ
            if not self.tree_selection_state.get(item, False):
                continue

            values = self.tree.item(item, 'values')
            if len(values) < 3:  # select, main, dup が必要
                continue

            main_path = Path(values[1])  # values[0]はチェックボックス
            duplicate_path = Path(values[2])

            # keep_sideに基づいてtargetとlinkを決定
            if keep_side == 'main':
                # αを残す：α→ターゲット、β→リンク
                target_path = main_path
                link_base = duplicate_path if name_mode == 'keep_compare' else duplicate_path.with_name(main_path.name)
                original_to_delete = duplicate_path
            else:
                # βを残す：β→ターゲット、α→リンク
                target_path = duplicate_path
                link_base = main_path if name_mode == 'keep_main' else main_path.with_name(duplicate_path.name)
                original_to_delete = main_path

            if not target_path.exists():
                fail += 1
                self.log_message(f'[FAIL] リンク先の元ファイルが存在しません: {target_path}')
                continue

            link_actual = link_base
            suffix = link_actual.suffix
            if suffix.lower() != '.lnk':
                link_actual = link_actual.with_suffix((suffix + '.lnk') if suffix else '.lnk')

            if link_actual.exists() or link_actual.is_symlink():
                removed, error = self._remove_existing_path(link_actual)
                if not removed:
                    fail += 1
                    self.log_message(f'[FAIL] 既存のショートカットを整理できませんでした: {link_actual} ({error})')
                    continue

            try:
                link_actual.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                fail += 1
                self.log_message(f'[FAIL] ショートカットの出力先を作成できませんでした: {link_actual.parent} ({exc})')
                continue

            # 相対パスを計算
            target_for_shortcut = target_path
            if use_relative_path:
                try:
                    target_for_shortcut = Path(os.path.relpath(target_path, link_actual.parent))
                except (ValueError, OSError):
                    # 異なるドライブなど相対パス化できない場合は絶対パスのまま
                    pass

            operations.append({'link': link_actual, 'target': target_for_shortcut})
            contexts.append((link_actual, target_for_shortcut, original_to_delete, target_path))

        message = ''
        batch_script = ''
        if operations and not cancelled:
            ok, message, batch_script = self._execute_shortcut_batch(operations)
            for link_path, target_for_sc, original_del, target_abs in contexts:
                link_str = str(link_path)
                target_str = str(target_for_sc)
                created = link_path.exists()
                fallback_msg = ''
                fallback_script = ''
                if not created:
                    fallback_entries = [(str(link_path), str(target_for_sc))]
                    fallback_script = _build_shortcut_batch_script(fallback_entries)
                    single_ok, single_msg = _run_shortcut_batch(fallback_entries, script=fallback_script)
                    fallback_msg = single_msg or ''
                    if single_ok:
                        time.sleep(0.05)
                        created = link_path.exists()
                if created:
                    success += 1
                    created_counts['ショートカット'] += 1
                    # パス形式を判定（相対パスかどうか）
                    is_relative = not Path(target_str).is_absolute() if target_str else False
                    path_type_label = ", 相対パス" if is_relative else ", 絶対パス"
                    # ログには絶対パスを表示
                    target_display = str(target_abs)
                    self.log_message(f'[OK] {link_str} -> {target_display} (ショートカット{path_type_label})')
                    actual_kind = self._classify_link_path(link_path)
                    if actual_kind not in ('実体', 'missing') and actual_kind != 'ショートカット':
                        self.log_message(f'[WARN] 実際のリンク種別が想定と異なります: 要求=ショートカット, 実際={actual_kind} ({link_str})')
                    removed, remove_error = self._remove_existing_path(original_del)
                    if not removed and remove_error:
                        self.log_message(f'[WARN] 元ファイルの削除に失敗しました: {original_del} ({remove_error})')
                else:
                    fail += 1
                    detail_parts = [message, fallback_msg]
                    detail = ' / '.join(part for part in detail_parts if part) or 'ショートカットの作成に失敗しました'
                    self.log_message(f'[FAIL] {link_str} -> {target_str} ({detail})')
            if message and ok:
                self.log_message(f'ショートカット作成処理からのメッセージ: {message}')
        elif cancelled and contexts:
            self.log_message('キャンセルしたため比較側の実体は削除されていません')
        elif message:
            self.log_message(f'[WARN] ショートカット作成処理からの出力: {message}')

        summary = f'リンク化処理完了: 成功 {success} 件, 失敗 {fail} 件'
        if cancelled:
            summary += ' (キャンセルされました)'
        self.log_message(summary)
        if created_counts:
            detail = ', '.join(f'{lt}: {cnt}件' for lt, cnt in created_counts.items())
            self.log_message(f'作成されたリンクの内訳: {detail}')

        main_value = self.main_dir.get()
        compare_value = self.compare_dir.get()
        realize_value = self.realize_source.get()
        self.root.after(0, lambda: self._post_linking_ui_updates(main_value, compare_value, realize_value))

    def _post_linking_ui_updates(self, main_path: str, compare_path: str, realize_path: str) -> None:
        self.link_btn.config(state='normal')
        self.detect_btn.config(state='normal')
        if main_path:
            self._schedule_link_summary(main_path, self.main_link_summary_var)
        if compare_path:
            self._schedule_link_summary(compare_path, self.compare_link_summary_var)
        if realize_path:
            self._schedule_link_summary(realize_path, self.realize_link_summary_var)

    def run_linking(self):
        items = self.tree.get_children()
        if not items:
            messagebox.showinfo('情報', '検出された重複がありません')
            return
        
        # 選択されているアイテム数を確認
        selected_count = sum(1 for item in items if self.tree_selection_state.get(item, False))
        if selected_count == 0:
            messagebox.showwarning('警告', 'リンク化するアイテムが選択されていません。\n左側のチェックボックスで対象を選択してください。')
            return
        
        link_type = self.link_type.get()
        name_mode = self.link_name_mode.get()
        path_mode = self.link_path_mode.get()
        keep_side = self.keep_side.get()
        
        mode_label = 'α側を残す' if name_mode == 'keep_main' else 'β側を残す'
        path_label = '相対パス' if path_mode == 'relative' else '絶対パス'
        keep_label = 'αを残す（β側をリンク化）' if keep_side == 'main' else 'βを残す（α側をリンク化）'
        
        self.log_message(f'リンク処理を開始: 種別={link_type}, ファイル名モード={mode_label}, パス形式={path_label}, 残す側={keep_label}, 選択数={selected_count}件')
        self.cancel_flag.clear()
        self.link_btn.config(state='disabled')
        self.detect_btn.config(state='disabled')
        threading.Thread(target=self._linking_worker, args=(link_type, name_mode, path_mode, keep_side), daemon=True).start()

    def _linking_worker(self, link_type: str, name_mode: str, path_mode: str, keep_side: str):
        use_relative_path = (path_mode == 'relative')
        
        if link_type == 'ショートカット':
            self._linking_shortcut_batch(name_mode, use_relative_path, keep_side)
            return

        success = fail = 0
        created_counts: Counter[str] = Counter()
        cancelled = False
        for item in self.tree.get_children():
            if self.cancel_flag.is_set():
                cancelled = True
                self.log_message('ユーザーがキャンセルしました')
                break

            # チェックボックスで選択されていないものはスキップ
            if not self.tree_selection_state.get(item, False):
                continue

            values = self.tree.item(item, 'values')
            if len(values) < 3:  # select, main, dup が必要
                continue

            main_path = Path(values[1])  # values[0]はチェックボックス
            duplicate_path = Path(values[2])

            # keep_sideに基づいてtargetとlinkを決定
            if keep_side == 'main':
                # αを残す：α→ターゲット、β→リンク
                target_path = main_path
                link_path = duplicate_path if name_mode == 'keep_compare' else duplicate_path.with_name(main_path.name)
            else:
                # βを残す：β→ターゲット、α→リンク
                target_path = duplicate_path
                link_path = main_path if name_mode == 'keep_main' else main_path.with_name(duplicate_path.name)

            if not target_path.exists():
                fail += 1
                self.log_message(f'[FAIL] リンク先の元ファイルが存在しません: {target_path}')
                continue

            if link_path == target_path:
                self.log_message(f'[SKIP] 対象とリンク先が同一のためスキップ: {target_path}')
                continue

            # 既存ファイルの削除処理
            if keep_side == 'main' and name_mode == 'keep_main' and link_path != duplicate_path:
                try:
                    if duplicate_path.exists() or duplicate_path.is_symlink():
                        if duplicate_path.is_dir() and not duplicate_path.is_symlink():
                            shutil.rmtree(duplicate_path)
                        else:
                            duplicate_path.unlink()
                except Exception as exc:
                    fail += 1
                    self.log_message(f'[FAIL] 元の比較側ファイルを整理できませんでした: {duplicate_path} ({exc})')
                    continue
            elif keep_side == 'compare' and name_mode == 'keep_compare' and link_path != main_path:
                try:
                    if main_path.exists() or main_path.is_symlink():
                        if main_path.is_dir() and not main_path.is_symlink():
                            shutil.rmtree(main_path)
                        else:
                            main_path.unlink()
                except Exception as exc:
                    fail += 1
                    self.log_message(f'[FAIL] 元のα側ファイルを整理できませんでした: {main_path} ({exc})')
                    continue

            ok, msg, actual_link_path = create_link(target_path, link_path, link_type, use_relative_path)
            if ok:
                success += 1
                created_counts[link_type] += 1
                link_display = actual_link_path if actual_link_path is not None else link_path
                link_str = str(link_display)
                target_str = str(target_path)
                
                # パス形式を表示
                path_type_label = f", {path_mode}パス" if link_type in ['シンボリックリンク', 'ショートカット'] else ""
                self.log_message(f'[OK] {link_str} -> {target_str} ({link_type}{path_type_label})')
                
                actual_kind = self._classify_link_path(Path(link_display))
                if actual_kind not in ('実体', 'missing') and actual_kind != link_type:
                    self.log_message(f'[WARN] 実際のリンク種別が想定と異なります: 要求={link_type}, 実際={actual_kind} ({link_str})')
            else:
                fail += 1
                error_desc = self._describe_link_error(msg)
                fail_link = str(link_path)
                fail_target = str(target_path)
                self.log_message(f'[FAIL] {fail_link} -> {fail_target} ({error_desc})')

        summary = f'リンク化処理完了: 成功 {success} 件, 失敗 {fail} 件'
        if cancelled:
            summary += ' (キャンセルされました)'
        self.log_message(summary)
        if created_counts:
            detail = ', '.join(f'{lt}: {cnt}件' for lt, cnt in created_counts.items())
            self.log_message(f'作成されたリンクの内訳: {detail}')

        main_value = self.main_dir.get()
        compare_value = self.compare_dir.get()
        realize_value = self.realize_source.get()
        self.root.after(0, lambda: self._post_linking_ui_updates(main_value, compare_value, realize_value))

    def cancel_process(self):
        self.cancel_flag.set()

    def open_csv(self):
        p = Path('duplicate_report.csv')
        if p.exists(): os.startfile(str(p.resolve()))
        else: messagebox.showinfo('情報', 'レポートが見つかりません')

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
