"""Synchronous subprocess execution and private process observations."""

from __future__ import annotations

import subprocess
import time
from typing import Literal

from black_box_optimizer.models import CandidateConfiguration, WorkerSpec


_TERMINATION_GRACE_SECONDS = 2.0
_ERROR_MESSAGE_LIMIT = 1_000
ExecutionStatus = Literal[
    "completed",
    "process_failed",
    "timed_out",
    "launch_failed",
    "cancelled",
]


def execute(
    worker_spec: WorkerSpec,
    candidate: CandidateConfiguration,
    metrics_path,
) -> dict[str, object]:
    """Run one worker synchronously and return private observations."""
    command = _build_command(worker_spec, candidate, metrics_path)
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None

    try:

        process = subprocess.Popen(
            command,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except KeyboardInterrupt:
        return _result(
            started,
            exit_code=None,
            timed_out=False,
            execution_status="cancelled",
            error_message="Worker cancelled by user before launch completed",
        )
    except OSError as error:
        message = _bounded_message("Worker launch failed", str(error))
        return _result(
            started,
            exit_code=None,
            timed_out=False,
            execution_status="launch_failed",
            error_message=message,
        )

    try:
        stdout, stderr = process.communicate(
            timeout=worker_spec.timeout_seconds
        )

    except subprocess.TimeoutExpired as error:
        stdout, stderr = _terminate_and_collect(process)
        stdout = stdout or _output_text(error.output)
        stderr = stderr or _output_text(error.stderr)
        message = _bounded_message(
            f"Worker exceeded {worker_spec.timeout_seconds}-second timeout",
            _last_nonblank_line(stderr),
        )
        return _result(
            started,
            exit_code=None,
            timed_out=True,
            execution_status="timed_out",
            error_message=message,
            stdout=stdout,
            stderr=stderr,
        )
    except KeyboardInterrupt:
        stdout, stderr = _terminate_and_collect(process)
        message = _bounded_message(
            "Worker cancelled by user",
            _last_nonblank_line(stderr),
        )
        return _result(
            started,
            exit_code=process.returncode,
            timed_out=False,
            execution_status="cancelled",
            error_message=message,
            stdout=stdout,
            stderr=stderr,
        )

    if process.returncode == 0:
        return _result(
            started,
            exit_code=0,
            timed_out=False,
            execution_status="completed",
            error_message=None,
            stdout=stdout,
            stderr=stderr,
        )

    message = _bounded_message(
        f"Worker exited with code {process.returncode}",
        _last_nonblank_line(stderr),
    )
    return _result(
        started,
        exit_code=process.returncode,
        timed_out=False,
        execution_status="process_failed",
        error_message=message,
        stdout=stdout,
        stderr=stderr,
    )


def _build_command(
    worker_spec: WorkerSpec,
    candidate: CandidateConfiguration,
    metrics_path,
) -> list[str]:
    command = list(worker_spec.command)
    for name, value in candidate.parameters.items():
        command.extend((f"--{name.replace('_', '-')}", str(value)))
    command.extend((worker_spec.metrics_argument, str(metrics_path)))
    return command


def _terminate_and_collect(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    process.terminate()
    try:
        stdout, stderr = process.communicate(
            timeout=_TERMINATION_GRACE_SECONDS
        )
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    return _output_text(stdout), _output_text(stderr)


def _last_nonblank_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _bounded_message(prefix: str, detail: str) -> str:
    if not detail:
        return prefix[:_ERROR_MESSAGE_LIMIT]
    separator = ": "
    available = _ERROR_MESSAGE_LIMIT - len(prefix) - len(separator)
    if available <= 0:
        return prefix[:_ERROR_MESSAGE_LIMIT]
    return f"{prefix}{separator}{detail[-available:]}"


def _output_text(output: object) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def _result(
    started: float,
    *,
    exit_code: int | None,
    timed_out: bool,
    execution_status: ExecutionStatus,
    error_message: str | None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, object]:
    return {
        "runtime_seconds": time.perf_counter() - started,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_status": execution_status,
        "error_message": error_message,
        "stdout": stdout,
        "stderr": stderr,
    }
