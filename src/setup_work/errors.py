"""Error types and the process exit codes they map to.

Exit codes are part of the public interface (scripts depend on them):

    0  success
    1  generic failure (an operation ran and failed)
    2  invalid usage or invalid configuration
    3  a required external dependency is missing
    4  the requested configuration is unsupported on this machine
    5  aborted by the user (confirmation declined)
    6  requested object not found (model, harness, server)
    7  a check did not pass (doctor / harness-doctor / serve-test / config-validate)
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    USAGE = 2
    MISSING_DEPENDENCY = 3
    UNSUPPORTED = 4
    ABORTED = 5
    NOT_FOUND = 6
    CHECK_FAILED = 7


class SetupWorkError(Exception):
    """Base error. `hint` is an actionable next step shown to the user."""

    exit_code: ExitCode = ExitCode.FAILURE

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message + (f"\nhint: {self.hint}" if self.hint else "")


class ConfigError(SetupWorkError):
    exit_code = ExitCode.USAGE


class UsageError(SetupWorkError):
    exit_code = ExitCode.USAGE


class MissingDependencyError(SetupWorkError):
    exit_code = ExitCode.MISSING_DEPENDENCY


class UnsupportedError(SetupWorkError):
    exit_code = ExitCode.UNSUPPORTED


class AbortedError(SetupWorkError):
    exit_code = ExitCode.ABORTED


class NotFoundError(SetupWorkError):
    exit_code = ExitCode.NOT_FOUND


class CheckFailedError(SetupWorkError):
    exit_code = ExitCode.CHECK_FAILED
