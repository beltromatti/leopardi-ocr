from __future__ import annotations

import re
from dataclasses import dataclass

from leopardi.inference.config import InferenceStageConfig, ValidationConfig
from leopardi.schemas.output import ParsedPage


@dataclass(slots=True)
class ValidationFinding:
    severity: str
    code: str
    message: str


@dataclass(slots=True)
class PageValidationReport:
    valid: bool
    error_count: int
    warning_count: int
    findings: tuple[ValidationFinding, ...]


def _unescaped_dollar_count(markdown: str) -> int:
    count = 0
    escaped = False
    for char in markdown:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "$":
            count += 1
    return count


def _table_findings(markdown: str) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    lines = markdown.splitlines()
    contiguous: list[str] = []
    for line in lines + [""]:
        if "|" in line and line.strip():
            contiguous.append(line)
            continue
        if len(contiguous) >= 2:
            counts = [row.count("|") for row in contiguous if row.strip()]
            if counts and min(counts) != max(counts):
                findings.append(
                    ValidationFinding(
                        severity="warning",
                        code="table_shape_inconsistent",
                        message="Table row pipe counts are inconsistent.",
                    )
                )
        contiguous = []
    return findings


def validate_markdown(markdown: str, config: ValidationConfig) -> PageValidationReport:
    findings: list[ValidationFinding] = []
    if config.require_balanced_code_fences and markdown.count("```") % 2 != 0:
        findings.append(
            ValidationFinding(
                severity="error",
                code="unbalanced_code_fence",
                message="Markdown code fences are not balanced.",
            )
        )
    if config.require_balanced_math_delimiters:
        if markdown.count("$$") % 2 != 0:
            findings.append(
                ValidationFinding(
                    severity="error",
                    code="unbalanced_display_math",
                    message="Display math delimiters are not balanced.",
                )
            )
        if _unescaped_dollar_count(re.sub(r"\$\$", "", markdown)) % 2 != 0:
            findings.append(
                ValidationFinding(
                    severity="warning",
                    code="unbalanced_inline_math",
                    message="Inline math delimiters may be unbalanced.",
                )
            )
    if config.require_table_shape_checks:
        findings.extend(_table_findings(markdown))
    if not config.allow_html_tables and "<table" in markdown.lower():
        findings.append(
            ValidationFinding(
                severity="error",
                code="html_table_disallowed",
                message="HTML tables are disallowed by the inference policy.",
            )
        )
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return PageValidationReport(
        valid=error_count == 0,
        error_count=error_count,
        warning_count=warning_count,
        findings=tuple(findings),
    )


def validate_parsed_page(page: ParsedPage, stage: InferenceStageConfig) -> PageValidationReport:
    findings: list[ValidationFinding] = []
    if not page.blocks:
        findings.append(
            ValidationFinding(
                severity="error",
                code="empty_blocks",
                message="Parsed page contains no blocks.",
            )
        )
    report = validate_markdown(page.markdown, stage.validation)
    findings.extend(report.findings)
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return PageValidationReport(
        valid=error_count == 0,
        error_count=error_count,
        warning_count=warning_count,
        findings=tuple(findings),
    )


def repair_required(report: PageValidationReport, stage: InferenceStageConfig) -> bool:
    return report.error_count >= stage.validation.max_error_count_before_hard_fail or not report.valid
