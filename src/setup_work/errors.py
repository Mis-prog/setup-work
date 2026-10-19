"""Ошибки и соответствующие коды завершения процесса.

Коды являются частью интерфейса для скриптов:
0 — успешно; 1 — ошибка операции; 2 — неверные параметры или конфигурация;
3 — отсутствует зависимость; 4 — конфигурация не поддерживается;
5 — отмена пользователем; 6 — объект не найден; 7 — проверка не пройдена."""

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
    """Базовая ошибка; hint содержит конкретную подсказку пользователю."""

    exit_code: ExitCode = ExitCode.FAILURE

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message + (f"\nчто сделать: {self.hint}" if self.hint else "")


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
