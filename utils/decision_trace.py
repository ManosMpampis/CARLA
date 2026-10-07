"""Export an interactive replay of live anomaly decisions."""

import base64
import json
from pathlib import Path


def render_decision_trace_fragment(trace: dict) -> str:
    """Embed a numeric RunningScorer trace in the reusable graph fragment."""
    if trace.get("threshold") is None:
        raise ValueError("decision trace needs a calibrated threshold")
    if not trace.get("updates"):
        raise ValueError("decision trace has no window updates")
    source = Path(__file__).with_name("decision_trace_fragment.html").read_text()
    data = json.dumps(trace, allow_nan=False).encode("utf-8")
    encoded = base64.b64encode(data).decode("ascii")
    return source.replace("__TRACE_BASE64__", encoded, 1)


def save_decision_trace_html(trace: dict, path) -> None:
    """Write a self-contained, browser-openable decision replay."""
    fragment = render_decision_trace_fragment(trace)
    html = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decision dynamics</title>
<style>
  :root { color-scheme: light dark;
    --background: light-dark(#ffffff, #14171a);
    --foreground: light-dark(#20252b, #edf1f3);
    --muted-foreground: light-dark(#606a73, #aeb6bc);
    --border: light-dark(#d5dbe0, #465057);
    --viz-series-1: light-dark(#2d64b2, #8ab4f8);
    --viz-series-2: light-dark(#a87810, #f1c45b);
    --viz-series-3: light-dark(#297d55, #6bd4a0);
    --viz-series-4: light-dark(#ba3c4d, #ff8491);
  }
  body { margin: 0; padding: 18px; background: var(--background);
    color: var(--foreground); font: 14px system-ui, sans-serif; }
  main { max-width: 920px; margin: 0 auto; }
  input { color: var(--foreground); background: var(--background); }
  input[type="range"] { width: 100%; }
  input[type="number"] { padding: 5px; border: 1px solid var(--border); }
</style>
</head>
<body><main>''' + fragment + '''</main></body></html>'''
    Path(path).write_text(html)
