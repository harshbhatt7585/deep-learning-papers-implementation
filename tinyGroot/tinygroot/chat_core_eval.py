"""ChatCORE evaluation: matches nanochat's chat-format eval suite.

Tasks: ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee.
Categorical tasks score by argmax over allowed letter tokens at the
<|assistant_start|> position. Generative tasks sample with the chat
template, with calculator tool use for <|python_start|>...<|python_end|>
spans (GSM8K, SpellingBee). HumanEval runs generated programs in a
sandboxed subprocess. The final ChatCORE metric is the random-baseline
centered mean accuracy across all six tasks.
"""

from __future__ import annotations

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import re
import signal
import tempfile
import warnings
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Optional


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_gsm_answer(completion: str) -> Optional[str]:
    match = GSM_RE.search(completion)
    if match:
        answer = match.group(1).strip().replace(",", "")
        try:
            number = Decimal(answer)
            if number == number.to_integral_value():
                return format(number, "f").split(".", 1)[0]
            return format(number.normalize(), "f")
        except DecimalException:
            return answer
    return None


def extract_imports(prompt: str) -> str:
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def extract_program(completion: str) -> str:
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


@contextlib.contextmanager
def _calc_timeout(seconds: int, formula: str):
    def handler(signum, frame):
        raise TimeoutError(f"'{formula}' timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _eval_with_timeout(formula: str, max_time: int = 3):
    try:
        with _calc_timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        return None


def use_calculator(expr: str):
    """Tool-use evaluator: safe math, and string .count() for SpellingBee."""
    expr = expr.replace(",", "")
    if all(c in "0123456789*+-/.() " for c in expr):
        if "**" in expr:
            return None
        return _eval_with_timeout(expr)
    allowed = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789'\"()._ "
    )
    if not all(c in allowed for c in expr):
        return None
    dangerous = ("__", "import", "exec", "eval", "compile", "open", "file",
                 "input", "globals", "locals", "vars", "dir",
                 "getattr", "setattr", "delattr", "hasattr")
    low = expr.lower()
    if any(p in low for p in dangerous):
        return None
    if ".count(" not in expr:
        return None
    return _eval_with_timeout(expr)


class _TimeoutException(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **k): raise IOError
    def readline(self, *a, **k): raise IOError
    def readlines(self, *a, **k): raise IOError
    def readable(self, *a, **k): return False


class _redirect_stdin(contextlib._RedirectStream):  # type: ignore[misc]
    _stream = "stdin"


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    timeout: bool = False


@contextlib.contextmanager
def _exec_time_limit(seconds: float):
    def handler(signum, frame):
        raise _TimeoutException("Timed out")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _chdir_ctx(root: str):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as d:
        with _chdir_ctx(d):
            yield d


@contextlib.contextmanager
def _capture_io():
    out = io.StringIO()
    err = io.StringIO()
    block = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(out):
        with contextlib.redirect_stderr(err):
            with _redirect_stdin(block):
                yield out, err


def _reliability_guard(max_memory: Optional[int]):
    if platform.uname().system != "Darwin" and max_memory is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_DATA, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_STACK, (max_memory, max_memory))
    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    import subprocess
    subprocess.Popen = None  # type: ignore[assignment]
    import sys
    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[mod] = None


def _unsafe_execute(code: str, timeout: float, max_memory: Optional[int], result):
    with _create_tempdir():
        import os as _os
        import shutil as _shutil
        rmtree, rmdir_, chdir_, unlink = _shutil.rmtree, _os.rmdir, _os.chdir, _os.unlink
        _reliability_guard(max_memory)
        result.update({"success": False, "stdout": "", "stderr": "", "timeout": False, "error": None})
        try:
            with _capture_io() as (so, se):
                with _exec_time_limit(timeout):
                    exec(code, {})
            result.update({"success": True, "stdout": so.getvalue(), "stderr": se.getvalue()})
        except _TimeoutException:
            result.update({"timeout": True, "error": "timeout"})
        except BaseException as e:
            result.update({"error": f"{type(e).__name__}: {e}"})
        _shutil.rmtree, _os.rmdir, _os.chdir, _os.unlink = rmtree, rmdir_, chdir_, unlink


def execute_code(code: str, timeout: float = 5.0, max_memory: Optional[int] = 256 * 1024 * 1024) -> ExecutionResult:
    """Run untrusted code in a subprocess sandbox. Returns success/stdout/stderr."""
    manager = multiprocessing.Manager()
    result = manager.dict()
    p = multiprocessing.Process(target=_unsafe_execute, args=(code, timeout, max_memory, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        return ExecutionResult(success=False, error="killed", timeout=True)
    if not result:
        return ExecutionResult(success=False, error="no result", timeout=True)
    return ExecutionResult(
        success=bool(result["success"]),
        stdout=result["stdout"],
        stderr=result["stderr"],
        error=result["error"],
        timeout=bool(result["timeout"]),
    )
