"""Check repository source files for monolith hygiene violations.

The hard verification rule is the configured physical-line limit. Preferred
line length is advisory because generated syntax, URLs, and other legitimate
cases can make an 80-character limit impractical. Oversized-file exceptions
must be exact, explicit, and approved by a human.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Sequence, TextIO


CONFIG_NAME = "source_hygiene.json"
REQUIRED_EXCEPTION_FIELDS = {
    "path",
    "approved_by",
    "approved_on",
    "coherent_responsibility",
    "why_splitting_is_worse",
}
DISALLOWED_APPROVER_NAMES = {
    "agent",
    "ai",
    "automated",
    "codex",
    "none",
    "tbd",
}


class ConfigurationError(ValueError):
    """Raised when source-hygiene configuration is invalid."""


@dataclass(frozen=True)
class ApprovedException:
    """A human-approved exception for one exact source path."""

    path: str
    approved_by: str
    approved_on: date
    coherent_responsibility: str
    why_splitting_is_worse: str


@dataclass(frozen=True)
class HygieneConfiguration:
    """Validated checker settings."""

    max_physical_lines: int
    preferred_line_length: int
    source_extensions: frozenset[str]
    exclude_paths: tuple[str, ...]
    approved_exceptions: tuple[ApprovedException, ...]


@dataclass(frozen=True)
class SourceFileReport:
    """Line-count observations for one source file."""

    path: str
    physical_lines: int
    long_line_numbers: tuple[int, ...]
    approved_exception: ApprovedException | None


@dataclass(frozen=True)
class HygieneReport:
    """Complete repository scan result."""

    files: tuple[SourceFileReport, ...]
    read_errors: tuple[str, ...]
    stale_exception_paths: tuple[str, ...]
    max_physical_lines: int
    preferred_line_length: int

    @property
    def unapproved_oversized_files(self) -> tuple[SourceFileReport, ...]:
        """Return oversized files that lack a valid exception."""

        return tuple(
            report
            for report in self.files
            if report.physical_lines > self.max_physical_lines
            and report.approved_exception is None
        )

    @property
    def approved_oversized_files(self) -> tuple[SourceFileReport, ...]:
        """Return oversized files covered by valid human approvals."""

        return tuple(
            report
            for report in self.files
            if report.physical_lines > self.max_physical_lines
            and report.approved_exception is not None
        )

    @property
    def passed(self) -> bool:
        """Whether hard verification requirements passed."""

        return not self.unapproved_oversized_files and not self.read_errors


def _require_positive_integer(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{key!r} must be a positive integer")
    return value


def _require_string_list(data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{key!r} must be a nonempty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigurationError(
            f"every entry in {key!r} must be a nonempty string"
        )
    return tuple(item.strip() for item in value)


def _normalize_relative_path(raw_path: str) -> str:
    normalized = raw_path.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(
            f"exception path must be repository-relative: {raw_path!r}"
        )
    if any(character in normalized for character in "*?[]"):
        raise ConfigurationError(
            f"exception path must identify one exact file: {raw_path!r}"
        )
    return path.as_posix()


def _require_explanation(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or len(value.strip()) < 20:
        raise ConfigurationError(
            f"approved exception {key!r} must contain a written explanation"
        )
    return value.strip()


def _parse_exception(raw: object, index: int) -> ApprovedException:
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"approved_exceptions[{index}] must be an object"
        )
    unknown = set(raw) - REQUIRED_EXCEPTION_FIELDS
    missing = REQUIRED_EXCEPTION_FIELDS - set(raw)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ConfigurationError(
            f"approved_exceptions[{index}] has " + "; ".join(details)
        )

    approved_by = raw["approved_by"]
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ConfigurationError("approved_by must name the human approver")
    if approved_by.strip().casefold() in DISALLOWED_APPROVER_NAMES:
        raise ConfigurationError(
            f"approved_by must be a human name, not {approved_by!r}"
        )

    approved_on = raw["approved_on"]
    if not isinstance(approved_on, str):
        raise ConfigurationError("approved_on must use YYYY-MM-DD")
    try:
        parsed_date = date.fromisoformat(approved_on)
    except ValueError as error:
        raise ConfigurationError("approved_on must use YYYY-MM-DD") from error

    path_value = raw["path"]
    if not isinstance(path_value, str):
        raise ConfigurationError("approved exception path must be a string")

    return ApprovedException(
        path=_normalize_relative_path(path_value),
        approved_by=approved_by.strip(),
        approved_on=parsed_date,
        coherent_responsibility=_require_explanation(
            raw, "coherent_responsibility"
        ),
        why_splitting_is_worse=_require_explanation(
            raw, "why_splitting_is_worse"
        ),
    )


def load_configuration(path: Path) -> HygieneConfiguration:
    """Load and validate the JSON checker configuration."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        message = f"configuration file not found: {path}"
        raise ConfigurationError(message) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"could not read configuration file {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ConfigurationError("configuration root must be an object")
    allowed_keys = {
        "max_physical_lines",
        "preferred_line_length",
        "source_extensions",
        "exclude_paths",
        "approved_exceptions",
    }
    unknown_keys = set(data) - allowed_keys
    missing_keys = allowed_keys - set(data)
    if unknown_keys or missing_keys:
        details = []
        if missing_keys:
            details.append(f"missing {sorted(missing_keys)}")
        if unknown_keys:
            details.append(f"unknown {sorted(unknown_keys)}")
        raise ConfigurationError("invalid configuration: " + "; ".join(details))

    extensions = _require_string_list(data, "source_extensions")
    normalized_extensions = []
    for extension in extensions:
        has_separator = "/" in extension or "\\" in extension
        if not extension.startswith(".") or has_separator:
            raise ConfigurationError(
                f"invalid source extension: {extension!r}"
            )
        normalized_extensions.append(extension.casefold())

    exclude_paths = _require_string_list(data, "exclude_paths")
    raw_exceptions = data["approved_exceptions"]
    if not isinstance(raw_exceptions, list):
        raise ConfigurationError("approved_exceptions must be a list")
    exceptions = tuple(
        _parse_exception(raw, index)
        for index, raw in enumerate(raw_exceptions)
    )
    exception_keys = [exception.path.casefold() for exception in exceptions]
    if len(exception_keys) != len(set(exception_keys)):
        raise ConfigurationError("approved exception paths must be unique")

    return HygieneConfiguration(
        max_physical_lines=_require_positive_integer(
            data, "max_physical_lines"
        ),
        preferred_line_length=_require_positive_integer(
            data, "preferred_line_length"
        ),
        source_extensions=frozenset(normalized_extensions),
        exclude_paths=tuple(
            pattern.strip().replace("\\", "/")
            for pattern in exclude_paths
        ),
        approved_exceptions=exceptions,
    )


def _is_excluded(relative_path: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/**"):
            prefix = pattern[:-3].rstrip("/")
            is_below_prefix = relative_path.startswith(prefix + "/")
            if relative_path == prefix or is_below_prefix:
                return True
        if fnmatch.fnmatchcase(relative_path, pattern):
            return True
    return False


def _source_paths(
    root: Path,
    configuration: HygieneConfiguration,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root):
        current_path = Path(current_root)
        kept_directories = []
        for directory_name in sorted(directory_names):
            directory_path = current_path / directory_name
            relative = directory_path.relative_to(root).as_posix()
            if not _is_excluded(relative, configuration.exclude_paths):
                kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in sorted(file_names):
            path = current_path / file_name
            relative = path.relative_to(root).as_posix()
            if _is_excluded(relative, configuration.exclude_paths):
                continue
            if path.suffix.casefold() in configuration.source_extensions:
                paths.append(path)
    return tuple(sorted(paths, key=lambda path: path.as_posix().casefold()))


def check_repository(
    root: Path,
    configuration: HygieneConfiguration,
) -> HygieneReport:
    """Inspect configured source files below ``root``."""

    resolved_root = root.resolve()
    exception_map = {
        exception.path.casefold(): exception
        for exception in configuration.approved_exceptions
    }
    reports: list[SourceFileReport] = []
    read_errors: list[str] = []
    seen_paths: set[str] = set()

    for path in _source_paths(resolved_root, configuration):
        relative = path.relative_to(resolved_root).as_posix()
        seen_paths.add(relative.casefold())
        try:
            physical_lines = 0
            long_lines: list[int] = []
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                for physical_lines, line in enumerate(source, start=1):
                    if (
                        len(line.rstrip("\r\n"))
                        > configuration.preferred_line_length
                    ):
                        long_lines.append(physical_lines)
        except (OSError, UnicodeError) as error:
            read_errors.append(f"{relative}: {error}")
            continue

        reports.append(
            SourceFileReport(
                path=relative,
                physical_lines=physical_lines,
                long_line_numbers=tuple(long_lines),
                approved_exception=exception_map.get(relative.casefold()),
            )
        )

    stale_exceptions = tuple(
        exception.path
        for exception in configuration.approved_exceptions
        if exception.path.casefold() not in seen_paths
    )
    return HygieneReport(
        files=tuple(reports),
        read_errors=tuple(read_errors),
        stale_exception_paths=stale_exceptions,
        max_physical_lines=configuration.max_physical_lines,
        preferred_line_length=configuration.preferred_line_length,
    )


def write_report(report: HygieneReport, output: TextIO) -> None:
    """Write a clear human-readable verification report."""

    state = "PASS" if report.passed else "FAIL"
    print(f"Source hygiene check: {state}", file=output)
    print(f"Scanned source files: {len(report.files)}", file=output)
    print(
        f"Maximum physical lines: {report.max_physical_lines:,}",
        file=output,
    )

    if report.unapproved_oversized_files:
        print("\nUnapproved oversized source files:", file=output)
        for file_report in report.unapproved_oversized_files:
            print(
                f"  - {file_report.path}: "
                f"{file_report.physical_lines:,} lines",
                file=output,
            )
        print(
            "Add an exact human-approved exception or split along real "
            "responsibility boundaries.",
            file=output,
        )

    if report.approved_oversized_files:
        print("\nApproved oversized source files:", file=output)
        for file_report in report.approved_oversized_files:
            exception = file_report.approved_exception
            assert exception is not None
            print(
                f"  - {file_report.path}: "
                f"{file_report.physical_lines:,} lines; approved by "
                f"{exception.approved_by} on {exception.approved_on}",
                file=output,
            )

    if report.read_errors:
        print("\nSource files that could not be checked:", file=output)
        for error in report.read_errors:
            print(f"  - {error}", file=output)

    long_line_reports = tuple(
        file_report
        for file_report in report.files
        if file_report.long_line_numbers
    )
    if long_line_reports:
        print(
            f"\nLine-length advisories (over "
            f"{report.preferred_line_length} characters):",
            file=output,
        )
        for file_report in long_line_reports:
            shown = ", ".join(
                str(line_number)
                for line_number in file_report.long_line_numbers[:10]
            )
            remainder = len(file_report.long_line_numbers) - 10
            suffix = f" (+{remainder} more)" if remainder > 0 else ""
            print(
                f"  - {file_report.path}: lines {shown}{suffix}",
                file=output,
            )

    if report.stale_exception_paths:
        print("\nStale approved-exception entries:", file=output)
        for path in report.stale_exception_paths:
            print(f"  - {path}", file=output)

    print(
        "\nReminder: passing the line-count check does not prove that a "
        "collection of files is well separated.",
        file=output,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check global source-file monolith hygiene."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root; defaults to the parent of tools/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=f"configuration path; defaults to ROOT/{CONFIG_NAME}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    arguments = _build_parser().parse_args(argv)
    root = arguments.root.resolve()
    config_path = arguments.config or root / CONFIG_NAME
    if not config_path.is_absolute():
        config_path = root / config_path

    try:
        configuration = load_configuration(config_path)
        report = check_repository(root, configuration)
    except ConfigurationError as error:
        print(f"Source hygiene configuration error: {error}", file=sys.stderr)
        return 1

    write_report(report, sys.stdout)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
