"""
Indian banking business-day calendar for T+2 float calculations.

Rules (from plan.md §Operational Robustness §2, §CategorizeException §1):
  • Sundays are non-business days.
  • 2nd and 4th Saturdays of each month are non-business days.
  • RBI-declared national holidays are non-business days.
  • All other days are business days.

Usage:
    from app.utils.calendar import business_days_between
    biz_days = business_days_between(date(2026, 8, 18), date(2026, 8, 20))
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Set

# ══════════════════════════════════════════════════════════════
#  RBI HOLIDAY LIST (2026 — extensible per year)
# ══════════════════════════════════════════════════════════════

# Key national holidays observed across all RBI-regulated banks.
# This list should be updated annually.  Holidays that fall on
# a Sunday are already excluded by the Sunday rule.
_RBI_HOLIDAYS: Set[date] = {
    # 2026 major RBI holidays (approximate — gazette-dependent)
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 10),   # Maha Shivaratri (approx)
    date(2026, 3, 17),   # Holi (approx)
    date(2026, 3, 31),   # Id-ul-Fitr (approx)
    date(2026, 4, 2),    # Ram Navami (approx)
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # May Day
    date(2026, 5, 26),   # Buddha Purnima (approx)
    date(2026, 6, 7),    # Eid-ul-Adha (approx)
    date(2026, 7, 6),    # Muharram (approx)
    date(2026, 8, 15),   # Independence Day
    date(2026, 9, 4),    # Milad-un-Nabi (approx)
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra (approx)
    date(2026, 11, 9),   # Diwali (approx)
    date(2026, 11, 10),  # Diwali (approx)
    date(2026, 11, 30),  # Guru Nanak Jayanti (approx)
    date(2026, 12, 25),  # Christmas
}


def _is_2nd_or_4th_saturday(d: date) -> bool:
    """Check if a date is the 2nd or 4th Saturday of its month."""
    if d.weekday() != 5:  # Not a Saturday
        return False
    # Which Saturday of the month?  (day-1)//7 + 1 gives the ordinal.
    ordinal = (d.day - 1) // 7 + 1
    return ordinal in (2, 4)


def is_business_day(d: date) -> bool:
    """Return True if ``d`` is a business day under Indian banking rules."""
    # Sunday
    if d.weekday() == 6:
        return False
    # 2nd/4th Saturday
    if _is_2nd_or_4th_saturday(d):
        return False
    # RBI holiday
    if d in _RBI_HOLIDAYS:
        return False
    return True


def business_days_between(start: date, end: date) -> int:
    """Count business days between two dates (exclusive of start, inclusive of end).

    If ``end <= start``, returns 0.

    This matches the Indian banking convention for settlement float:
    T+0 = order day (not counted), T+1 = first business day, etc.
    """
    if end <= start:
        return 0

    count = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if is_business_day(cursor):
            count += 1
        cursor += timedelta(days=1)
    return count
