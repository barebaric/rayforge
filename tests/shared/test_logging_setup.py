import logging

from rayforge.logging_setup import UILogFilter


def _make_record(category: str | None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test",
        args=(),
        exc_info=None,
    )
    if category is not None:
        record.__dict__["log_category"] = category
    return record


def test_ui_categories_always_pass():
    log_filter = UILogFilter()
    for category in UILogFilter.UI_CATEGORIES:
        assert log_filter.filter(_make_record(category))


def test_verbose_categories_pass():
    log_filter = UILogFilter()
    for category in UILogFilter.VERBOSE_CATEGORIES:
        assert log_filter.filter(_make_record(category))


def test_raw_io_is_verbose_category():
    assert "RAW_IO" in UILogFilter.VERBOSE_CATEGORIES


def test_uncategorized_records_are_dropped():
    log_filter = UILogFilter()
    assert not log_filter.filter(_make_record(None))
