from rayforge.machine.sanity import (
    CheckMode,
    IssueCategory,
    IssueSeverity,
    SanityIssue,
    SanityReport,
)


def test_clean_report():
    report = SanityReport(mode=CheckMode.FAST)
    assert report.is_clean
    assert not report.has_errors
    assert not report.has_warnings


def test_report_with_error():
    report = SanityReport(
        mode=CheckMode.FAST,
        issues=[
            SanityIssue(
                category=IssueCategory.MACHINE_EXTENT,
                severity=IssueSeverity.ERROR,
                message="out of bounds",
            )
        ],
    )
    assert not report.is_clean
    assert report.has_errors
    assert not report.has_warnings


def test_report_with_warning():
    report = SanityReport(
        mode=CheckMode.FAST,
        issues=[
            SanityIssue(
                category=IssueCategory.WORKAREA,
                severity=IssueSeverity.WARNING,
                message="outside workarea",
            )
        ],
    )
    assert not report.is_clean
    assert not report.has_errors
    assert report.has_warnings


def test_report_mixed():
    report = SanityReport(
        mode=CheckMode.COMPLETE,
        issues=[
            SanityIssue(
                category=IssueCategory.MACHINE_EXTENT,
                severity=IssueSeverity.ERROR,
                message="err",
            ),
            SanityIssue(
                category=IssueCategory.WORKAREA,
                severity=IssueSeverity.WARNING,
                message="warn",
            ),
        ],
    )
    assert not report.is_clean
    assert report.has_errors
    assert report.has_warnings
