from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_report(
    path: Path, stats: dict[str, Any], duplicates: list[dict[str, str]] | None = None
) -> None:
    duplicates = duplicates or []

    def table(items: dict[str, Any]) -> str:
        return (
            "".join(
                f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
                for key, value in items.items()
            )
            or "<tr><td>No data</td></tr>"
        )

    summary = {
        key.replace("_", " ").title(): value
        for key, value in stats.items()
        if key not in {"resolutions", "formats", "objects_per_class", "issues"}
    }
    issues = "".join(
        f"<tr><td>{html.escape(issue['path'])}</td><td>{html.escape(issue['issue'])}</td></tr>"
        for issue in stats.get("issues", [])[:500]
    ) or "<tr><td colspan=\"2\">No issues</td></tr>"
    document = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Dataset report</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;color:#1f2937}}section{{margin:2rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border-bottom:1px solid #d1d5db;text-align:left}}th{{width:45%}}h1{{color:#111827}}</style></head><body><h1>Dataset report</h1><section><h2>Summary</h2><table>{table(summary)}</table></section><section><h2>Resolution distribution</h2><table>{table(stats.get("resolutions", {}))}</table></section><section><h2>File formats</h2><table>{table(stats.get("formats", {}))}</table></section><section><h2>Objects per class</h2><table>{table(stats.get("objects_per_class", {}))}</table></section><section><h2>Integrity issues</h2><table><tr><th>Path</th><th>Issue</th></tr>{issues}</table></section><section><h2>Duplicate summary</h2><p>{len(duplicates)} duplicate image(s) removed or flagged during build.</p></section></body></html>"""
    path.write_text(document, encoding="utf-8")
