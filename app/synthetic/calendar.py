"""
Re-export shim: ``app.synthetic.calendar`` → ``app.utils.calendar``.

The canonical implementation lives in ``app/utils/calendar.py``.
This module exists solely so existing imports like::

    from app.synthetic.calendar import business_days_between

continue to work without modification.
"""
from app.utils.calendar import (  # noqa: F401 — re-export
    business_days_between,
    is_business_day,
)
