from dataclasses import dataclass
from pathlib import Path

from .reports import list_reports
from .settings import ProviderSettings, ResearchSettings
from .standalone.llm import redact_text


@dataclass(frozen=True)
class LocalSetupDiagnostics:
    """Secret-free local setup diagnosis for user-facing surfaces."""

    ok: bool
    missing_required: list[str]
    provider_summary: str
    reports_dir: Path
    report_count: int
    error_message: str = ""


def diagnose_local_setup(provider: ProviderSettings, research: ResearchSettings) -> LocalSetupDiagnostics:
    """Check local provider settings and report storage without constructing runtime."""

    missing = provider.missing_required()
    report_count = 0
    error_message = ""
    try:
        report_count = len(list_reports(research.reports_dir))
    except Exception as error:
        error_message = redact_text(str(error))
    return LocalSetupDiagnostics(
        ok=not missing and not error_message,
        missing_required=missing,
        provider_summary=provider.safe_summary(),
        reports_dir=research.reports_dir,
        report_count=report_count,
        error_message=error_message,
    )


def format_diagnostics_text(diagnostics: LocalSetupDiagnostics, *, success_message: str) -> str:
    """Format a local setup diagnosis without exposing credential values."""

    if diagnostics.ok:
        lines = [success_message]
    elif diagnostics.missing_required:
        lines = [f"配置预检未通过：缺少 {', '.join(diagnostics.missing_required)}"]
    else:
        lines = ["配置预检未通过。"]
    lines.extend(
        [
            f"Provider: {diagnostics.provider_summary}",
            f"Reports dir: {diagnostics.reports_dir}",
            f"Local reports: {diagnostics.report_count}",
        ]
    )
    if diagnostics.error_message:
        lines.append(f"诊断错误：{diagnostics.error_message}")
    return "\n".join(lines)
