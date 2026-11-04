"""Campaign reports.

The point of a report is not to restate the metrics: it is to say what was
measured, under which conditions, and what the run cannot prove. A number
without its corpus hash, its n and its blind spots is a claim, not a result.
"""

from norn.report.campaign import (
    Report,
    build_report,
    render_html,
    render_markdown,
    render_png,
    write_report,
)

__all__ = [
    "Report",
    "build_report",
    "render_html",
    "render_markdown",
    "render_png",
    "write_report",
]
