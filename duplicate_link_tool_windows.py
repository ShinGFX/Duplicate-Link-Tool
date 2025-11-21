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
ASSET_ICON_NAME = 'assets/duplicate_link_tool_from256.ico'


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
        lines.append(f"    $lnk = $shell.CreateShortcut('{escaped_link}')")
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
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.ps1', encoding='utf-8') as tmp:
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

        name = os.path.basename(path)
        primary_candidates = list(main_lookup.get((name, size), []))
        for candidate in primary_candidates:
            if cancel_flag.is_set():
                return None
            if not os.path.exists(candidate):
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
            tree.insert('', 'end', values=r)
        count_label.config(
            text=(
                f'重複検出数: {len(results)} 件 / '
                f'非重複(メイン): {len(main_non_duplicates)} 件 / '
                f'非重複(比較): {len(compare_non_duplicates)} 件'
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
                    f'非重複(メイン) {len(main_non_duplicates)} 件 / '
                    f'非重複(比較) {len(compare_non_duplicates)} 件'
                )
                log_callback(log_text)
            except Exception:
                pass

    tk_root.after(0, apply_results)

    with open('duplicate_report.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Main','Duplicate','Reason'])
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
    if mode == 'junction':
        cmd = f'mklink /J "{link_path}" "{target_path}"'
    elif mode == 'symbolic_dir':
        cmd = f'mklink /D "{link_path}" "{target_path}"'
    else:
        cmd = f'mklink "{link_path}" "{target_path}"'
    try:
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def create_link(target_path: Path, link_path: Path, link_type: str) -> tuple[bool, str, Optional[Path]]:
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
            ok, msg = run_mklink(str(link_actual), str(target), mode=mode)
            if not ok:
                return False, msg, None
        elif link_type == 'ショートカット':
            ok, msg = create_shortcut(link_actual, target)
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
                root.iconbitmap(default=str(icon_path))
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
        self.link_type = tk.StringVar(value='シンボリックリンク')
        self.link_name_mode = tk.StringVar(value='keep_compare')
        self.cancel_flag = threading.Event()
        self.compare_non_duplicates = []
        self.main_non_duplicates = []
        self.main_link_summary_var = tk.StringVar(value='リンク状況: 未選択')
        self.compare_link_summary_var = tk.StringVar(value='リンク状況: 未選択')
        self.realize_link_summary_var = tk.StringVar(value='リンク状況: 未選択')

        top = tk.Frame(root, bg=DARK_BG)
        top.pack(padx=10, pady=10, fill='x')
        top.columnconfigure(0, weight=1)

        tk.Label(top, text='1) メインディレクトリ (残す側)', bg=DARK_BG, fg=DARK_FG).grid(row=0, column=0, sticky='w')
        tk.Entry(top, textvariable=self.main_dir, width=80, bg='#2b2b2b', fg=DARK_FG).grid(row=1, column=0, sticky='we')
        tk.Button(top, text='選択', command=self.select_main).grid(row=1, column=1, padx=5)
        tk.Label(top, textvariable=self.main_link_summary_var, bg=DARK_BG, fg=DARK_FG).grid(row=2, column=0, columnspan=2, sticky='w', pady=(2,0))

        tk.Label(top, text='2) 比較ディレクトリ (削除してリンク化する側)', bg=DARK_BG, fg=DARK_FG).grid(row=3, column=0, sticky='w', pady=(8,0))
        tk.Entry(top, textvariable=self.compare_dir, width=80, bg='#2b2b2b', fg=DARK_FG).grid(row=4, column=0, sticky='we')
        tk.Button(top, text='選択', command=self.select_compare).grid(row=4, column=1, padx=5)
        tk.Label(top, textvariable=self.compare_link_summary_var, bg=DARK_BG, fg=DARK_FG).grid(row=5, column=0, columnspan=2, sticky='w', pady=(2,0))

        tk.Label(top, text='3) 実体化して圧縮するディレクトリ', bg=DARK_BG, fg=DARK_FG).grid(row=6, column=0, sticky='w', pady=(8,0))
        tk.Entry(top, textvariable=self.realize_source, width=80, bg='#2b2b2b', fg=DARK_FG).grid(row=7, column=0, sticky='we')
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

        tk.Label(top, text='リンク作成後に残すファイル名', bg=DARK_BG, fg=DARK_FG).grid(row=11, column=0, sticky='w', pady=(10,0))
        name_frame = tk.Frame(top, bg=DARK_BG)
        name_frame.grid(row=12, column=0, columnspan=2, sticky='w')
        tk.Radiobutton(
            name_frame,
            text='比較側の名前を残す',
            variable=self.link_name_mode,
            value='keep_compare',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)
        tk.Radiobutton(
            name_frame,
            text='メイン側の名前を残す',
            variable=self.link_name_mode,
            value='keep_main',
            bg=DARK_BG,
            fg=DARK_FG,
            selectcolor=DARK_BG,
        ).pack(side='left', padx=4)

        btn_frame = tk.Frame(root, bg=DARK_BG)
        btn_frame.pack(pady=6)
        self.detect_btn = tk.Button(btn_frame, text='重複を検出', command=self.detect)
        self.detect_btn.pack(side='left', padx=6)
        self.link_btn = tk.Button(btn_frame, text='リンク化を実行', command=self.run_linking)
        self.link_btn.pack(side='left', padx=6)
        self.cancel_btn = tk.Button(btn_frame, text='キャンセル', command=self.cancel_process)
        self.cancel_btn.pack(side='left', padx=6)
        self.open_main_nondup_btn = tk.Button(
            btn_frame,
            text='非重複(メイン)を開く',
            command=lambda: self.open_non_duplicates('main'),
            state='disabled',
        )
        self.open_main_nondup_btn.pack(side='left', padx=6)
        self.open_compare_nondup_btn = tk.Button(
            btn_frame,
            text='非重複(比較)を開く',
            command=lambda: self.open_non_duplicates('compare'),
            state='disabled',
        )
        self.open_compare_nondup_btn.pack(side='left', padx=6)
        self.materialize_btn = tk.Button(btn_frame, text='リンクを実体化', command=self.materialize_links)
        self.materialize_btn.pack(side='left', padx=6)
        self.realize_btn = tk.Button(btn_frame, text='実体化して圧縮', command=self.realize_and_zip)
        self.realize_btn.pack(side='left', padx=6)
        tk.Button(btn_frame, text='CSV を開く', command=self.open_csv).pack(side='left', padx=6)

        self.progress = ttk.Progressbar(root, length=860, mode='determinate', style='Horizontal.TProgressbar')
        self.progress.pack(pady=8)
        self.count_label = tk.Label(
            root,
            text='重複検出数: 0 件 / 非重複(メイン): 0 件 / 非重複(比較): 0 件',
            bg=DARK_BG,
            fg=DARK_FG,
        )
        self.count_label.pack()
        self.total_size_var = tk.StringVar(value='重複合計サイズ: 0 B')
        self.total_size_label = tk.Label(root, textvariable=self.total_size_var, bg=DARK_BG, fg=DARK_FG)
        self.total_size_label.pack()
        self.zip_status_var = tk.StringVar(value='ステータス: 待機中')
        self.zip_status_label = tk.Label(root, textvariable=self.zip_status_var, bg=DARK_BG, fg=DARK_FG)
        self.zip_status_label.pack()

        tree_container = tk.Frame(root, bg=DARK_BG)
        tree_container.pack(fill='both', expand=True, padx=10, pady=(6, 3))

        tree_frame = tk.Frame(tree_container, bg=DARK_BG)
        tree_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=('main', 'dup', 'reason'), show='headings', height=13, selectmode='browse')
        self.tree.heading('main', text='Main')
        self.tree.heading('dup', text='Duplicate')
        self.tree.heading('reason', text='Reason')
        self.tree.column('main', width=360)
        self.tree.column('dup', width=360)
        self.tree.column('reason', width=200)
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        tree_scroll_y.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True, side='left')

        log_container = tk.Frame(root, bg=DARK_BG, height=180)
        log_container.pack(fill='x', padx=10, pady=(0, 10))
        log_container.pack_propagate(False)

        tk.Label(log_container, text='ログ', bg=DARK_BG, fg=DARK_FG, anchor='w').pack(fill='x')
        log_frame = tk.Frame(log_container, bg=DARK_BG)
        log_frame.pack(fill='both', expand=True)
        self.log = tk.Text(log_frame, bg='#2b2b2b', fg=DARK_FG, wrap='word')
        log_scroll_y = ttk.Scrollbar(log_frame, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll_y.set)
        log_scroll_y.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True, side='left')

        self.tree.bind('<ButtonRelease-1>', self.on_tree_click)
        self._adjust_initial_geometry()
        self.root.after(200, self._ensure_front)

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
            self.root.update_idletasks()
            required_w = max(1000, self.root.winfo_reqwidth())
            required_h = max(850, self.root.winfo_reqheight())
            self.root.geometry(f'{required_w}x{required_h}')
        except Exception:
            pass

    def _format_link_summary(self, counts: dict[str, int]) -> str:
        parts = []
        if counts.get('reparse'):
            parts.append(f"シンボリック/ジャンクション {counts['reparse']}件")
        if counts.get('hardlink'):
            parts.append(f"ハードリンク {counts['hardlink']}件")
        if counts.get('shortcut'):
            parts.append(f"ショートカット {counts['shortcut']}件")
        if not parts:
            return 'リンク状況: リンクなし'
        return 'リンク状況: ' + ', '.join(parts)

    def _link_summary_worker(self, root_path: Path, var: tk.StringVar) -> None:
        counts = {'reparse': 0, 'hardlink': 0, 'shortcut': 0}
        try:
            for dirpath, dirnames, filenames in os.walk(root_path):
                for name in dirnames + filenames:
                    path = Path(dirpath) / name
                    try:
                        st = os.lstat(path)
                    except Exception:
                        continue
                    if path.suffix.lower() == '.lnk':
                        counts['shortcut'] += 1
                        continue
                    if stat.S_ISLNK(st.st_mode):
                        counts['reparse'] += 1
                        continue
                    if not stat.S_ISDIR(st.st_mode) and st.st_nlink > 1:
                        counts['hardlink'] += 1
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
            self.log_message(f'メインディレクトリ: {d}')
            self._schedule_link_summary(d, self.main_link_summary_var)

    def select_compare(self):
        d = filedialog.askdirectory()
        if d:
            self.compare_dir.set(d)
            self.log_message(f'比較ディレクトリ: {d}')
            self._schedule_link_summary(d, self.compare_link_summary_var)

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
        self.log.delete('1.0', 'end')
        self.progress['value'] = 0
        self.cancel_flag.clear()
        self.compare_non_duplicates.clear()
        self.main_non_duplicates.clear()
        self.open_compare_nondup_btn.config(state='disabled')
        self.open_main_nondup_btn.config(state='disabled')
        self.count_label.config(
            text='重複検出数: 集計中... / 非重複(メイン): 集計中... / 非重複(比較): 集計中...'
        )
        self.zip_status_var.set('ステータス: 重複検出を準備しています...')
        self.total_size_var.set('重複合計サイズ: 計算中...')
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

    def on_tree_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        self.tree.selection_set(item_id)
        self.tree.focus(item_id)
        # 少し遅延させて選択状態の更新後に処理
        self.root.after(50, lambda: self._open_pair_from_item(item_id))

    def _open_pair_from_item(self, item_id):
        values = self.tree.item(item_id, 'values')
        if len(values) < 2:
            return
        main_path, duplicate_path = values[0], values[1]
        opened_any = False

        for label, path in (('Main', main_path), ('Duplicate', duplicate_path)):
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
        label = 'メイン' if side == 'main' else '比較'

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

        try:
            for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
                base = Path(dirpath)
                for name in filenames:
                    handled, ok, out_path, message = self._materialize_entry(base / name)
                    if not handled:
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

    def _linking_shortcut_batch(self, name_mode: str) -> None:
        success = fail = 0
        created_counts: Counter[str] = Counter()
        cancelled = False
        operations: list[dict[str, Path]] = []
        contexts: list[tuple[Path, Path, Path]] = []

        for item in self.tree.get_children():
            if self.cancel_flag.is_set():
                cancelled = True
                self.log_message('ユーザーがキャンセルしました')
                break

            values = self.tree.item(item, 'values')
            if len(values) < 2:
                continue

            main_path = Path(values[0])
            duplicate_path = Path(values[1])

            if not main_path.exists():
                fail += 1
                self.log_message(f'[FAIL] リンク先の元ファイルが存在しません: {main_path}')
                continue

            target_path = main_path
            link_base = duplicate_path if name_mode == 'keep_compare' else duplicate_path.with_name(main_path.name)

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

            operations.append({'link': link_actual, 'target': target_path})
            contexts.append((link_actual, target_path, duplicate_path))

        message = ''
        batch_script = ''
        if operations and not cancelled:
            ok, message, batch_script = self._execute_shortcut_batch(operations)
            for link_path, target_path, duplicate_path in contexts:
                link_str = str(link_path)
                target_str = str(target_path)
                created = link_path.exists()
                fallback_msg = ''
                fallback_script = ''
                if not created:
                    fallback_entries = [(str(link_path), str(target_path))]
                    fallback_script = _build_shortcut_batch_script(fallback_entries)
                    single_ok, single_msg = _run_shortcut_batch(fallback_entries, script=fallback_script)
                    fallback_msg = single_msg or ''
                    if single_ok:
                        time.sleep(0.05)
                        created = link_path.exists()
                        if created and fallback_msg:
                            self.log_message(f'[INFO] 再試行でショートカット作成に成功: {link_str} ({fallback_msg})')
                if created:
                    success += 1
                    created_counts['ショートカット'] += 1
                    self.log_message(f'[OK] {link_str} -> {target_str} (ショートカット)')
                    actual_kind = self._classify_link_path(link_path)
                    if actual_kind not in ('実体', 'missing') and actual_kind != 'ショートカット':
                        self.log_message(f'[WARN] 実際のリンク種別が想定と異なります: 要求=ショートカット, 実際={actual_kind} ({link_str})')
                    removed, remove_error = self._remove_existing_path(duplicate_path)
                    if not removed and remove_error:
                        self.log_message(f'[WARN] 比較側の削除に失敗しました: {duplicate_path} ({remove_error})')
                else:
                    fail += 1
                    detail_parts = [message, fallback_msg]
                    detail = ' / '.join(part for part in detail_parts if part) or 'ショートカットの作成に失敗しました'
                    self.log_message(f'[FAIL] {link_str} -> {target_str} ({detail})')
                    if fallback_script:
                        snippet_single = fallback_script if len(fallback_script) < 2000 else fallback_script[:2000] + '...'
                        self.log_message('[DEBUG] PowerShellバッチスクリプト(単体):')
                        self.log_message(snippet_single)
            if not ok and batch_script:
                snippet = batch_script if len(batch_script) < 2000 else batch_script[:2000] + '...'
                self.log_message('[DEBUG] PowerShellバッチスクリプト:')
                self.log_message(snippet)
            if message and ok:
                self.log_message(f'[INFO] ショートカット作成処理からのメッセージ: {message}')
        elif cancelled and contexts:
            self.log_message('[INFO] キャンセルしたため比較側の実体は削除されていません')
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
        link_type = self.link_type.get()
        name_mode = self.link_name_mode.get()
        mode_label = 'メイン側を残す' if name_mode == 'keep_main' else '比較側を残す'
        self.log_message(f'リンク処理を開始: 種別={link_type}, ファイル名モード={mode_label}')
        self.cancel_flag.clear()
        self.link_btn.config(state='disabled')
        self.detect_btn.config(state='disabled')
        threading.Thread(target=self._linking_worker, args=(link_type, name_mode), daemon=True).start()

    def _linking_worker(self, link_type: str, name_mode: str):
        if link_type == 'ショートカット':
            self._linking_shortcut_batch(name_mode)
            return

        success = fail = 0
        created_counts: Counter[str] = Counter()
        cancelled = False
        for item in self.tree.get_children():
            if self.cancel_flag.is_set():
                cancelled = True
                self.log_message('ユーザーがキャンセルしました')
                break

            values = self.tree.item(item, 'values')
            if len(values) < 2:
                continue

            main_path = Path(values[0])
            duplicate_path = Path(values[1])

            target_path = main_path
            link_path = duplicate_path if name_mode == 'keep_compare' else duplicate_path.with_name(main_path.name)

            if not target_path.exists():
                fail += 1
                self.log_message(f'[FAIL] リンク先の元ファイルが存在しません: {target_path}')
                continue

            if link_path == target_path:
                self.log_message(f'[SKIP] 対象とリンク先が同一のためスキップ: {target_path}')
                continue

            if name_mode == 'keep_main' and link_path != duplicate_path:
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

            ok, msg, actual_link_path = create_link(target_path, link_path, link_type)
            if ok:
                success += 1
                created_counts[link_type] += 1
                link_display = actual_link_path if actual_link_path is not None else link_path
                link_str = str(link_display)
                target_str = str(target_path)
                self.log_message(f'[OK] {link_str} -> {target_str} ({link_type})')
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
    root = tk.Tk(); app = App(root); root.mainloop()
