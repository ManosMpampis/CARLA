"""Export an interactive replay of live anomaly decisions."""

import base64
import json
from pathlib import Path


def write_final_decision_trace(model, device, series, config, threshold, *,
                               options=None, score_transform=None,
                               threshold_operator="ge",
                               score_label="Mean score") -> dict | None:
    """Replay a bounded prefix of final scoring and write its decision graph.

    The default 256-window limit keeps a normal score run from embedding a
    multi-gigabyte history in HTML. Set decision_trace.max_windows to null
    to replay the complete evaluation window sequence.
    """
    from utils.scoring import RunningScorer, evaluation_options

    setting = config.get("decision_trace", {})
    if setting is False:
        return None
    if setting is None or setting is True:
        setting = {}
    if not isinstance(setting, dict):
        raise ValueError("decision_trace must be a YAML mapping or false")
    limit = setting.get("max_windows", 256)
    if limit is not None:
        if isinstance(limit, bool) or int(limit) != limit or int(limit) < 1:
            raise ValueError("decision_trace.max_windows must be positive or null")
        limit = int(limit)
    options = dict(evaluation_options(config) if options is None else options)
    window, stride = int(options.pop("wsz")), int(options.pop("stride"))
    if len(series) < window:
        raise ValueError("series is shorter than the decision trace input window")
    starts = list(range(0, len(series) - window + 1, stride))
    if options.get("include_last_window", stride < window) and starts[-1] != len(series) - window:
        starts.append(len(series) - window)
    selected = starts if limit is None else starts[:limit]
    running = RunningScorer(
        model, device, window, check_mask=options.get("check_mask"),
        latency_offset=options.get("latency_offset", 0), threshold=threshold,
        record_history=True)#, threshold_operator=threshold_operator, score_transform=score_transform, series_length=len(series))
    for start in selected:
        running.update(series[start:start + window], start)
    trace = running.decision_trace()
    trace["score_label"] = score_label
    trace["scope"] = {
        "recorded_windows": len(selected), "total_windows": len(starts),
        "first_input_start": selected[0], "last_input_start": selected[-1],
        "truncated": len(selected) < len(starts),
    }
    directory = Path(config["jepa_dir"])
    html_path = directory / "decision_trace.html"
    json_path = directory / "decision_trace.json"
    save_decision_trace_html(trace, html_path)
    json_path.write_text(json.dumps(trace, indent=2, allow_nan=False))
    return {"html": str(html_path), "json": str(json_path), **trace["scope"]}


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
