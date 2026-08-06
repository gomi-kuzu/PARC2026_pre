from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


MAX_ZIP_SIZE_MB = 20 * 1024
MAX_EXTRACTED_SIZE_MB = 40 * 1024
MAX_ENTRIES = 200_000
MAX_COMPRESSION_RATIO = 300

REQUIRED_FILES = ["policy_server.py", "requirements.txt"]
REQUIRED_ENDPOINTS = ["/health", "/reset", "/act"]
REQUIRED_METHODS = ["get_action", "reset"]


JUNK_COMPONENTS = {".git", "__pycache__", ".venv", "venv", "node_modules",
                   ".DS_Store", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}


BANNED_REQ_OPTIONS = ("-i", "--index-url", "--extra-index-url", "-f", "--find-links",
                      "-e", "--editable", "-r", "--requirement", "-c", "--constraint")


SAFE_REQ_OPTIONS = ("--hash", "--require-hashes", "--no-index", "--pre",
                    "--no-binary", "--only-binary", "--prefer-binary")


BANNED_REQ_SCHEMES = ("file", "http", "https", "ftp", "ftps", "ssh", "git", "hg",
                      "svn", "bzr")

_SCHEME_RE = re.compile(r"(?:^|[@\s=(])([a-z][a-z0-9]*(?:\+[a-z0-9]+)?)\s*:", re.I)


DEFAULT_CAMERA = 128


ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


@dataclass
class Report:

    items: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str) -> None:
        self.items.append((severity, code, message))

    def error(self, code: str, message: str) -> None:
        self.add(ERROR, code, message)

    def warn(self, code: str, message: str) -> None:
        self.add(WARN, code, message)

    def info(self, code: str, message: str) -> None:
        self.add(INFO, code, message)

    def extend(self, other: "Report") -> None:
        self.items.extend(other.items)

    @property
    def errors(self) -> list[tuple[str, str, str]]:
        return [i for i in self.items if i[0] == ERROR]

    @property
    def warnings(self) -> list[tuple[str, str, str]]:
        return [i for i in self.items if i[0] == WARN]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [{"code": c, "message": m} for s, c, m in self.items if s == ERROR],
            "warnings": [{"code": c, "message": m} for s, c, m in self.items if s == WARN],
            "info": [{"code": c, "message": m} for s, c, m in self.items if s == INFO],
        }

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        n_e, n_w = len(self.errors), len(self.warnings)
        lines = [f"バリデーション結果: {status}  (errors={n_e}, warnings={n_w})"]
        icon = {ERROR: "✗", WARN: "!", INFO: "·"}
        for sev, code, msg in self.items:
            lines.append(f"  [{icon[sev]}] {sev:5s} {code}: {msg}")
        return "\n".join(lines)


def _find_root_prefix(names: list[str]) -> tuple[str | None, str | None]:
    files = [n for n in names if not n.endswith("/")]

    if any(n == "policy_server.py" for n in files):
        return "", None

    tops = set()
    for n in names:
        head = n.split("/", 1)[0]
        if head:
            tops.add(head)
    if len(tops) == 1:
        top = next(iter(tops))
        prefix = f"{top}/"
        if any(n == f"{prefix}policy_server.py" for n in files):
            note = (f"policy_server.py がルート直下でなく単一フォルダ '{top}/' の"
                    f"配下にあります（Finder/Explorer のフォルダ包み zip）。自動で"
                    f"1階層降りて評価しますが、採点環境によっては即エラーになるため、"
                    f"フォルダを含めず中身を直接 zip 化することを推奨します。")
            return prefix, note
    return None, None


def validate_zip(zip_path: str | Path, report: Report | None = None) -> Report:
    report = report or Report()
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        report.error("zip.missing", f"提出 zip が存在しません: {zip_path}")
        return report


    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_ZIP_SIZE_MB:
        report.error("zip.too_large",
                     f"zip サイズ超過: {size_mb:.1f}MB > 上限 {MAX_ZIP_SIZE_MB}MB (20GB)")
    elif size_mb > MAX_ZIP_SIZE_MB * 0.8:
        report.warn("zip.size_warn", f"zip サイズが上限の80%超: {size_mb:.1f}MB")


    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as e:
        report.error("zip.corrupt", f"zip として開けません（破損の可能性）: {e}")
        return report

    with zf:
        infos = zf.infolist()
        names = zf.namelist()


        if any(i.flag_bits & 0x1 for i in infos):
            report.error("zip.encrypted",
                         "暗号化（パスワード付き）zip は展開・評価できません")
            return report


        if len(infos) > MAX_ENTRIES:
            report.error("zip.too_many_entries",
                         f"エントリ数超過: {len(infos)} > 上限 {MAX_ENTRIES}")


        _check_zip_slip(infos, report)


        total_uncomp = sum(i.file_size for i in infos)
        total_comp = sum(i.compress_size for i in infos)
        uncomp_mb = total_uncomp / (1024 * 1024)
        if uncomp_mb > MAX_EXTRACTED_SIZE_MB:
            report.error("zip.bomb_size",
                         f"展開後合計サイズ超過: {uncomp_mb:.1f}MB > 上限 "
                         f"{MAX_EXTRACTED_SIZE_MB}MB。zip bomb か巨大な非モデル物の"
                         f"混入を疑ってください")
        if total_comp > 0:
            ratio = total_uncomp / total_comp
            if ratio > MAX_COMPRESSION_RATIO:
                report.error("zip.bomb_ratio",
                             f"圧縮率が異常: {ratio:.0f}x > 上限 {MAX_COMPRESSION_RATIO}x "
                             f"（zip bomb の疑い）")


        if not report.ok:
            return report


        try:
            bad = zf.testzip()
            if bad is not None:
                report.error("zip.corrupt", f"zip 内の壊れたエントリ: {bad}")
        except Exception as e:
            report.error("zip.corrupt", f"zip の検証に失敗（破損の可能性）: {e}")
            return report


        prefix, note = _find_root_prefix(names)
        if prefix is None:
            report.error("struct.no_policy_server",
                         "policy_server.py が見つかりません（ルート直下 or 単一"
                         "トップフォルダ配下のいずれにもありません）")
            return report
        if note:
            report.warn("struct.nested_root", note)

        present = set(names)
        for req in REQUIRED_FILES:
            if f"{prefix}{req}" not in present:
                sev = report.error if req == "policy_server.py" else report.warn
                sev(f"struct.missing_{req.split('.')[0]}",
                    f"必須ファイルが見つかりません: {req}")


        _check_junk(names, report)


        ps_name = f"{prefix}policy_server.py"
        if ps_name in present:
            try:
                src = zf.read(ps_name).decode("utf-8", errors="replace")
                _check_policy_server_source(src, report)
            except Exception as e:
                report.warn("policy.read_error", f"policy_server.py を読めません: {e}")

        req_name = f"{prefix}requirements.txt"
        if req_name in present:
            try:
                req_src = zf.read(req_name).decode("utf-8", errors="replace")
                check_requirements(req_src, report)
            except Exception as e:
                report.warn("req.read_error", f"requirements.txt を読めません: {e}")

    return report


def _check_zip_slip(infos: list[zipfile.ZipInfo], report: Report) -> None:
    for info in infos:
        name = info.filename

        if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
            report.error("zip.slip_absolute", f"絶対パスのエントリを拒否: {name}")

        parts = name.replace("\\", "/").split("/")
        if ".." in parts:
            report.error("zip.slip_traversal", f"'..' を含むパスを拒否: {name}")

        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            report.error("zip.slip_symlink", f"symlink エントリを拒否: {name}")


def _check_junk(names: list[str], report: Report) -> None:
    hit = set()
    for n in names:
        for comp in n.replace("\\", "/").split("/"):
            if comp in JUNK_COMPONENTS:
                hit.add(comp)
    for pat in sorted(hit):
        report.warn("struct.junk",
                    f"不要物 '{pat}' が含まれています（サイズ膨張の原因。"
                    f"提出前に除外を推奨）")


def _check_policy_server_source(src: str, report: Report) -> None:
    import ast


    src = src.lstrip("﻿")


    try:
        ast.parse(src)
    except SyntaxError as e:
        report.error("policy.syntax",
                     f"policy_server.py に構文エラー: line {e.lineno}: {e.msg}")
        return
    except ValueError as e:
        report.error("policy.syntax",
                     f"policy_server.py を解析できません（バイナリ/不正な文字の可能性）: {e}")
        return


    for ep in REQUIRED_ENDPOINTS:
        if f'"{ep}"' not in src and f"'{ep}'" not in src:
            report.warn("policy.missing_endpoint",
                        f"エンドポイント '{ep}' が見当たりません（サーバー部分を"
                        f"改変した可能性。評価クライアントはこのパスを叩きます）")


    for m in REQUIRED_METHODS:
        if f"def {m}" not in src:
            report.warn("policy.missing_method",
                        f"メソッド '{m}' の定義が見当たりません")


    if "--port" not in src:
        report.warn("policy.no_port_arg",
                    "'--port' 引数の処理が見当たりません。採点側は --port 8000 を"
                    "付けて起動するため、未対応だと別ポートで起動し health check が"
                    "タイムアウトする恐れがあります")


def _split_option(token: str) -> str:
    return token.split("=", 1)[0]


def check_requirements(text: str, report: Report) -> None:

    if text.startswith("﻿"):
        text = text[1:]


    logical: list[tuple[int, str]] = []
    buf = ""
    start_ln = 0
    for ln, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if not buf:
            start_ln = ln
        if line.endswith("\\"):
            buf += line[:-1]
            continue
        logical.append((start_ln, buf + line))
        buf = ""
    if buf:
        logical.append((start_ln, buf))

    try:
        from packaging.requirements import Requirement, InvalidRequirement
        have_packaging = True
    except Exception:
        have_packaging = False


    syntax_errors = 0

    for ln, line in logical:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        s = s.split(" #", 1)[0].strip()
        if not s:
            continue


        if s.startswith("-"):
            first = _split_option(s.split()[0])
            if first in BANNED_REQ_OPTIONS:
                report.error("req.external_option",
                             f"{ln}行目: 外部ソース指定 '{first}' は禁止です（install"
                             f"段階での通信・任意コード実行の温床）: {s}")
            elif first in SAFE_REQ_OPTIONS:
                pass
            else:
                report.error("req.unknown_option",
                             f"{ln}行目: 未知の pip オプション '{first}' は許可されて"
                             f"いません: {s}")
            continue


        core = re.split(r"\s(?=--(?:hash|require-hashes|global-option|config-settings)\b)",
                        s, maxsplit=1)[0].strip()


        banned = False
        for m in _SCHEME_RE.finditer(core):
            scheme = m.group(1).lower()
            base = scheme.split("+", 1)[0]
            if base in BANNED_REQ_SCHEMES or scheme in BANNED_REQ_SCHEMES:
                report.error("req.external_url",
                             f"{ln}行目: 外部ソース（スキーム '{scheme}:'）の指定は"
                             f"禁止です: {s}")
                banned = True
                break
        if banned:
            continue


        if core.startswith("./") or core.startswith("../") or core.startswith("/") \
                or core in (".", ".."):
            report.error("req.local_path",
                         f"{ln}行目: ローカルパス直接指定は禁止です（setup.py が install"
                         f"時に実行されます）: {s}")
            continue


        if have_packaging:
            try:
                parsed = Requirement(core)
            except InvalidRequirement as e:
                syntax_errors += 1
                if syntax_errors <= 10:
                    report.error("req.syntax",
                                 f"{ln}行目: 依存指定の構文エラー: {s} ({e})")
                elif syntax_errors == 11:
                    report.error("req.syntax",
                                 "構文エラーが多数（バイナリ/非テキストの requirements.txt "
                                 "の可能性）。以降は省略します")
                continue

            if parsed.url is not None:
                report.error("req.external_url",
                             f"{ln}行目: direct reference（URL 指定）は禁止です: {s}")


def pip_dry_run(req_path: str | Path, report: Report, timeout: int = 300) -> None:
    import subprocess
    req_path = Path(req_path)
    if not req_path.is_file():
        return
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "-q", "-r", str(req_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        report.error("req.dry_run_timeout",
                     f"pip の依存解決が {timeout}秒 以内に完了しませんでした（病的な"
                     f"バージョン制約の疑い）")
        return
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        report.error("req.dry_run_failed",
                     "pip の依存解決に失敗:\n      " + "\n      ".join(tail))


def safe_extract(zip_path: str | Path, dest_dir: str | Path,
                 report: Report | None = None) -> Path:
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise ValueError(f"symlink エントリは展開しません: {name}")
            target = (dest_dir / name).resolve()
            if not str(target).startswith(str(dest_dir) + os.sep) and target != dest_dir:
                raise ValueError(f"展開先を逸脱するエントリを拒否: {name}")
            zf.extract(info, dest_dir)


    names = zipfile.ZipFile(zip_path).namelist()
    prefix, _ = _find_root_prefix(names)
    root = dest_dir / prefix if prefix else dest_dir
    return root


def validate_dir(submission_dir: str | Path, report: Report | None = None) -> Report:
    report = report or Report()
    d = Path(submission_dir)
    if not d.is_dir():
        report.error("dir.missing", f"ディレクトリが存在しません: {d}")
        return report


    if not (d / "policy_server.py").is_file():
        entries = [p for p in d.iterdir() if p.name not in (".DS_Store",)]
        subdirs = [p for p in entries if p.is_dir()]
        if len(subdirs) == 1 and not any(p.is_file() for p in entries) \
                and (subdirs[0] / "policy_server.py").is_file():
            report.warn("struct.nested_root",
                        f"policy_server.py が単一フォルダ '{subdirs[0].name}/' の配下に"
                        f"あります。自動で1階層降りますが、zip 化時はフォルダを含めず"
                        f"中身を直接固めることを推奨します。")
            d = subdirs[0]

    ps = d / "policy_server.py"
    if not ps.is_file():
        report.error("struct.no_policy_server", "policy_server.py が見つかりません")
        return report
    if not (d / "requirements.txt").is_file():
        report.warn("struct.missing_requirements", "requirements.txt が見つかりません")


    total_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
    if total_mb > MAX_EXTRACTED_SIZE_MB:
        report.error("dir.too_large",
                     f"展開後合計サイズ超過: {total_mb:.1f}MB > 上限 {MAX_EXTRACTED_SIZE_MB}MB")

    _check_junk([str(p.relative_to(d)) for p in d.rglob("*")], report)

    try:
        _check_policy_server_source(ps.read_text(encoding="utf-8", errors="replace"), report)
    except Exception as e:
        report.warn("policy.read_error", f"policy_server.py を読めません: {e}")

    req = d / "requirements.txt"
    if req.is_file():
        check_requirements(req.read_text(encoding="utf-8", errors="replace"), report)

    return report


def _make_dummy_obs(camera: int):
    import numpy as np
    return {
        "agentview_image": np.zeros((camera, camera, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((camera, camera, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def _serialize_obs(obs) -> bytes:
    import msgpack
    serialized = {
        k: {"data": v.tobytes(), "shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in obs.items()
    }
    return msgpack.packb(serialized, use_bin_type=True)


def smoke_test(submission_dir: str | Path, report: Report | None = None, *,
               camera: int = DEFAULT_CAMERA, port: int = 8931,
               health_timeout: int = 180, act_timeout: float = 10.0,
               install: bool = False) -> Report:
    report = report or Report()
    d = Path(submission_dir)


    try:
        import numpy as np
        import msgpack
        import requests
    except Exception as e:
        report.warn("smoke.deps_missing",
                    f"動的スモークに必要な依存が無いためスキップ ({e})。"
                    f"`pip install numpy msgpack requests` で有効化されます")
        return report

    if not (d / "policy_server.py").is_file():
        report.error("smoke.no_server", "policy_server.py が無いため起動できません")
        return report

    import subprocess
    import time

    if install and (d / "requirements.txt").is_file():
        report.info("smoke.installing", "requirements.txt をインストール中...")
        pi = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=str(d), capture_output=True, text=True,
        )
        if pi.returncode != 0:
            tail = (pi.stderr or pi.stdout).strip().splitlines()[-50:]
            report.error("smoke.install_failed",
                         "依存インストールに失敗:\n      " + "\n      ".join(tail))
            return report

    log = tempfile.TemporaryFile(mode="w+b")


    proc = subprocess.Popen(
        [sys.executable, "policy_server.py", "--port", str(port)],
        cwd=str(d), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    descendants: list = []

    def _server_tail(n_bytes: int = 8192) -> str:

        try:
            log.flush()
            size = os.fstat(log.fileno()).st_size
            log.seek(max(0, size - n_bytes))
            data = log.read().decode("utf-8", errors="replace")
            return "\n      ".join(data.strip().splitlines()[-15:])
        except Exception:
            return "(ログ取得不可)"

    try:
        base = f"http://localhost:{port}"

        deadline = time.monotonic() + health_timeout
        healthy = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                report.error("smoke.server_died",
                             f"policy_server.py が起動直後に終了しました "
                             f"(exit={proc.returncode})。\n      {_server_tail()}")
                return report
            try:
                if requests.get(f"{base}/health", timeout=3).status_code == 200:
                    healthy = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        if not healthy:
            report.error("smoke.health_timeout",
                         f"/health が {health_timeout}秒以内に 200 を返しませんでした"
                         f"（重みロードが遅い/ポート不一致の疑い）。\n      {_server_tail()}")
            return report
        report.info("smoke.health_ok", "/health が 200 を返しました")


        try:
            import psutil
            descendants = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            descendants = []


        try:
            r = requests.post(f"{base}/reset",
                              json={"instruction": "pick up the object", "seed": 0},
                              timeout=act_timeout)
            if r.status_code != 200:
                report.error("smoke.reset_status",
                             f"/reset が {r.status_code} を返しました（毎エピソード"
                             f"呼ばれるため、失敗すると全エピソード0点）")
                return report
        except requests.RequestException as e:
            report.error("smoke.reset_error", f"/reset で例外: {e}")
            return report
        report.info("smoke.reset_ok", "/reset が 200 を返しました")


        obs = _make_dummy_obs(camera)
        payload = _serialize_obs(obs)
        latencies: list[float] = []
        first_action = None
        for i in range(3):
            t0 = time.monotonic()
            try:
                r = requests.post(f"{base}/act", data=payload,
                                  headers={"Content-Type": "application/x-msgpack"},
                                  timeout=act_timeout)
            except requests.Timeout:
                report.error("smoke.act_timeout",
                             f"/act が {act_timeout}秒以内に応答しませんでした")
                return report
            except requests.RequestException as e:
                report.error("smoke.act_error", f"/act で例外: {e}")
                return report
            latencies.append(time.monotonic() - t0)

            if r.status_code != 200:
                report.error("smoke.act_status", f"/act が {r.status_code} を返しました")
                return report

            action = _check_action_bytes(r.content, report, camera)
            if action is None:
                return report
            if i == 0:
                first_action = action


        import statistics
        mean_l, max_l = statistics.mean(latencies), max(latencies)
        report.info("smoke.latency",
                    f"/act レイテンシ: mean={mean_l:.3f}s max={max_l:.3f}s")
        if max_l > act_timeout:
            report.warn("smoke.slow",
                        f"/act が遅く、本番タイムアウト({act_timeout:.0f}s)に触れる恐れ")


        requests.post(f"{base}/reset",
                      json={"instruction": "pick up the object", "seed": 0},
                      timeout=act_timeout)
        r2 = requests.post(f"{base}/act", data=payload,
                           headers={"Content-Type": "application/x-msgpack"},
                           timeout=act_timeout)
        import numpy as np
        action2 = _check_action_bytes(r2.content, report, camera, quiet=True)
        if first_action is not None and action2 is not None:
            if np.allclose(first_action, action2, atol=1e-5):
                report.info("smoke.deterministic", "同一 seed/obs で action が再現しました")
            else:
                report.warn("smoke.nondeterministic",
                            "同一 seed/obs でも action が変化します（サンプリング等で"
                            "意図的なら問題なし。再現性が必要なら要確認）")
        return report
    finally:
        _kill_process_group(proc, descendants)
        log.close()


def _kill_process_group(proc, descendants=None) -> None:
    import signal
    import subprocess


    procs = list(descendants or [])
    try:
        import psutil
        try:
            parent = psutil.Process(proc.pid)
            procs += parent.children(recursive=True)
        except psutil.NoSuchProcess:
            pass

        try:
            procs.append(psutil.Process(proc.pid))
        except psutil.NoSuchProcess:
            pass

        uniq = {}
        for p in procs:
            try:
                uniq[p.pid] = p
            except Exception:
                pass
        for p in uniq.values():
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(uniq.values(), timeout=10)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        return
    except Exception:
        pass


    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _check_action_bytes(content: bytes, report: Report, camera: int,
                        quiet: bool = False):
    import msgpack
    import numpy as np
    try:
        unpacked = msgpack.unpackb(content, raw=False)
        data = unpacked["data"]
    except Exception as e:
        report.error("smoke.act_decode", f"/act の応答を msgpack デコードできません: {e}")
        return None

    if not isinstance(data, (bytes, bytearray)):
        report.error("smoke.act_format",
                     "/act 応答の 'data' が bytes ではありません（テンプレの "
                     "serialize_action を改変した可能性）")
        return None


    if len(data) % 4 != 0:
        report.error("smoke.act_bytes", f"action のバイト長が float32 の倍数でない: {len(data)}")
        return None
    action = np.frombuffer(data, dtype=np.float32)
    if action.shape != (7,):
        hint = ""


        if len(data) == 7 * 8:
            hint = ("（56バイト＝7要素を float64 で送っている可能性。action は "
                    "float32 で返してください）")
        elif len(data) == 7 * 2:
            hint = "（float16 で送っている可能性。float32 で返してください）"
        report.error("smoke.act_shape",
                     f"action shape が不正: 期待 (7,), 実際 {action.shape}。{hint}"
                     f"[dx,dy,dz,droll,dpitch,dyaw,gripper] の7次元 float32 を返してください")
        return None

    if not np.all(np.isfinite(action)):
        report.error("smoke.act_nan",
                     "action に NaN/Inf が含まれます（評価クライアントは shape しか"
                     "検査しないため素通りし、全タスク0点の最も調査しにくい失敗になります）")
        return None

    if not quiet:
        if np.any(np.abs(action[:6]) > 1.0):
            report.warn("smoke.act_range",
                        "action の一部が [-1,1] を逸脱しています（デルタ制御では"
                        "通常この範囲。前処理を確認してください）")
        report.info("smoke.act_ok", f"/act が有効な action を返しました: {np.round(action, 3)}")
    return action


def validate(target: str | Path, *, static_only: bool = False,
             camera: int = DEFAULT_CAMERA, install: bool = False,
             do_pip_dry_run: bool = False, port: int = 8931,
             health_timeout: int = 180) -> Report:
    report = Report()
    target = Path(target)

    tmp_root = None
    if target.is_file() and (zipfile.is_zipfile(target)
                             or target.suffix.lower() == ".zip"):
        validate_zip(target, report)
        if do_pip_dry_run and report.ok:

            with zipfile.ZipFile(target) as zf:
                prefix, _ = _find_root_prefix(zf.namelist())
                rn = f"{prefix}requirements.txt" if prefix is not None else None
                if rn and rn in zf.namelist():
                    tmp_root = Path(tempfile.mkdtemp(prefix="valreq_"))
                    (tmp_root / "requirements.txt").write_bytes(zf.read(rn))
                    pip_dry_run(tmp_root / "requirements.txt", report)
        if not report.ok:
            return report
        if not static_only:
            work = Path(tempfile.mkdtemp(prefix="valdyn_"))
            try:
                root = safe_extract(target, work, report)
                smoke_test(root, report, camera=camera, install=install,
                           port=port, health_timeout=health_timeout)
            finally:
                import shutil
                shutil.rmtree(work, ignore_errors=True)
    elif target.is_dir():
        validate_dir(target, report)
        if do_pip_dry_run:
            pip_dry_run(target / "requirements.txt", report)
        if report.ok and not static_only:
            smoke_test(target, report, camera=camera, install=install,
                       port=port, health_timeout=health_timeout)
    else:
        report.error("input.invalid", f"zip でもディレクトリでもありません: {target}")

    if tmp_root is not None:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VLA コンペ提出物のバリデータ（静的 + 動的スモーク）")
    parser.add_argument("target", help="提出 zip または展開済みディレクトリ")
    parser.add_argument("--static", action="store_true",
                        help="静的チェックのみ（サーバーを起動しない）")
    parser.add_argument("--install", action="store_true",
                        help="動的スモーク前に requirements.txt をインストールする")
    parser.add_argument("--pip-dry-run", action="store_true",
                        help="pip install --dry-run で依存解決可否を事前確認する（低速）")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA,
                        help=f"ダミー観測のカメラ解像度（既定 {DEFAULT_CAMERA}）")
    parser.add_argument("--port", type=int, default=8931,
                        help="スモークで使うポート（既定 8931）")
    parser.add_argument("--health-timeout", type=int, default=180,
                        help="/health 起動待ちの上限秒（既定 180）")
    parser.add_argument("--json", action="store_true",
                        help="結果を JSON で出力する")
    args = parser.parse_args(argv)

    report = validate(
        args.target, static_only=args.static, camera=args.camera,
        install=args.install, do_pip_dry_run=args.pip_dry_run,
        port=args.port, health_timeout=args.health_timeout,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
