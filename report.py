"""
HTML report generator for ASR Model Shootout.
Produces a self-contained, editorial-quality HTML file.
Designed to embed cleanly into ainewsumbrella articles.
"""

import json
import os
from datetime import datetime
from typing import Any

from config import PROVIDERS, PROJECTION_HOURS


def generate_report(results: dict[str, Any], output_path: str) -> str:
    html = _build_html(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Report saved: {output_path}")
    return output_path


def _build_html(results: dict) -> str:
    meta = results["metadata"]
    samples = results["samples"]
    summary = results["summary"]
    providers_tested = meta["providers_tested"]

    pcolors = {}
    pnames = {}
    for p in PROVIDERS:
        pcolors[p.id] = p.color
        pnames[p.id] = p.name

    chart_data = _build_chart_data(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASR Model Shootout — {meta['date']}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
{_css()}
</style>
</head>
<body>
<article class="report">

{_hero(meta, summary, pnames)}
{_leaderboard(summary, pnames, pcolors)}
{_radar_section()}
{_noise_section(samples, pnames, pcolors)}
{_speed_section(samples, pnames, pcolors, providers_tested, chart_data)}
{_accuracy_section(samples, pnames, pcolors, providers_tested)}
{_wer_chart_section()}
{_cost_section(pnames, pcolors, providers_tested)}
{_transcript_section(samples, pnames, pcolors, providers_tested)}
{_methodology(meta, samples)}
<footer class="footnote">
  Data collected {meta['date']}. {meta['total_api_calls']} API calls across {meta['runs']} runs per sample.
  Timing includes network latency. WER weighted by audio duration.
</footer>

</article>

<script>
const CHART_DATA = {json.dumps(chart_data)};
{_chart_js()}
</script>
</body>
</html>"""


# ── CSS ─────────────────────────────────────────────────────────────────────

def _css() -> str:
    return """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #ffffff;
  --bg-alt: #f8f9fa;
  --border: #e5e7eb;
  --text: #1a1a1a;
  --text-2: #4b5563;
  --text-3: #9ca3af;
  --accent: #0369a1;
  --accent-light: #e0f2fe;
  --red: #dc2626;
  --green: #16a34a;
  --serif: 'Source Serif 4', Georgia, serif;
  --sans: 'Inter', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111111;
    --bg-alt: #1a1a1a;
    --border: #2a2a2a;
    --text: #e5e5e5;
    --text-2: #a3a3a3;
    --text-3: #525252;
    --accent: #38bdf8;
    --accent-light: #0c2d48;
  }
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}

.report {
  max-width: 720px;
  margin: 0 auto;
  padding: 48px 20px 80px;
}

/* ── Typography ─── */
h1 {
  font-family: var(--serif);
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.02em;
  margin-bottom: 12px;
}
h2 {
  font-family: var(--serif);
  font-size: 1.35rem;
  font-weight: 600;
  margin: 48px 0 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
h3 {
  font-size: 0.9rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-2);
  margin: 24px 0 12px;
}
p, .prose {
  color: var(--text-2);
  font-size: 0.95rem;
  margin-bottom: 16px;
}

/* ── Hero ─── */
.hero {
  margin-bottom: 40px;
}
.hero .kicker {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
  margin-bottom: 8px;
}
.hero .dek {
  font-family: var(--serif);
  font-size: 1.15rem;
  color: var(--text-2);
  font-style: italic;
  margin-bottom: 24px;
}
.meta-line {
  font-size: 0.8rem;
  color: var(--text-3);
  border-top: 1px solid var(--border);
  padding-top: 12px;
}

/* ── Leaderboard ─── */
.leaderboard {
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin: 24px 0 32px;
}
.lb-row {
  display: grid;
  grid-template-columns: 32px 1fr 90px 90px 90px;
  gap: 0;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 0.88rem;
  align-items: center;
}
.lb-row:last-child { border-bottom: none; }
.lb-header {
  background: var(--bg-alt);
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-3);
  padding: 10px 16px;
}
.lb-rank {
  font-weight: 700;
  font-size: 1.1rem;
  color: var(--text-3);
}
.lb-name {
  font-weight: 600;
}
.lb-val {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.lb-best {
  color: var(--accent);
  font-weight: 600;
}

/* ── Table ─── */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  margin: 16px 0 24px;
  font-variant-numeric: tabular-nums;
}
.data-table th {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 2px solid var(--text);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-2);
}
.data-table th.num { text-align: right; }
.data-table td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}
.data-table td.num { text-align: right; }
.data-table tr:hover td { background: var(--bg-alt); }
.wer-good { color: var(--green); }
.wer-mid { color: #ca8a04; }
.wer-bad { color: var(--red); }

/* ── Chart ─── */
.chart-wrap {
  max-width: 100%;
  margin: 16px 0 24px;
  position: relative;
}

/* ── Noise comparison ─── */
.noise-grid {
  display: grid;
  grid-template-columns: 1fr 80px 80px 80px;
  gap: 0;
  font-size: 0.85rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin: 16px 0 24px;
}
.noise-grid .ng-cell {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.noise-grid .ng-cell:nth-child(4n+1) {
  text-align: left;
}
.noise-grid .ng-header {
  background: var(--bg-alt);
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-3);
}
.ng-arrow { color: var(--red); font-size: 0.8rem; }

/* ── Transcript diff ─── */
.transcript-tabs {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0;
}
.t-tab {
  background: none;
  border: none;
  padding: 8px 14px;
  font-size: 0.8rem;
  font-family: var(--sans);
  color: var(--text-3);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.t-tab:hover { color: var(--text); }
.t-tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  font-weight: 500;
}
.t-panel { display: none; }
.t-panel.active { display: block; }
.t-box {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px 16px;
  font-family: var(--mono);
  font-size: 0.82rem;
  line-height: 1.8;
  margin-bottom: 12px;
  overflow-x: auto;
}
.t-label {
  font-family: var(--sans);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 6px;
}
.word-match { color: var(--text); }
.word-error { color: var(--red); text-decoration: underline; text-decoration-style: wavy; text-underline-offset: 3px; }
.word-extra { color: #ca8a04; }
.word-missing { color: var(--text-3); text-decoration: line-through; }

/* ── Details ─── */
details {
  margin: 16px 0;
  border: 1px solid var(--border);
  border-radius: 4px;
}
details summary {
  padding: 12px 16px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  color: var(--text-2);
}
details summary:hover { color: var(--text); }
details .inner { padding: 0 16px 16px; }

/* ── Footer ─── */
.footnote {
  font-size: 0.78rem;
  color: var(--text-3);
  border-top: 1px solid var(--border);
  padding-top: 16px;
  margin-top: 48px;
  line-height: 1.6;
}

/* ── Responsive ─── */
@media (max-width: 640px) {
  h1 { font-size: 1.6rem; }
  .lb-row { grid-template-columns: 24px 1fr 70px 70px 70px; font-size: 0.78rem; padding: 10px 12px; }
  .noise-grid { grid-template-columns: 1fr 65px 65px 65px; font-size: 0.78rem; }
}
"""


# ── Sections ────────────────────────────────────────────────────────────────

def _hero(meta, summary, pnames) -> str:
    best_wer_id = min(summary, key=lambda k: summary[k].get("avg_wer", 999))
    best_speed_id = min(summary, key=lambda k: summary[k].get("avg_speed", 999))

    return f"""
<div class="hero">
  <div class="kicker">Benchmark</div>
  <h1>We tested four speech-to-text APIs on the same audio. Here's what we found.</h1>
  <p class="dek">Cohere Transcribe, OpenAI Whisper, Deepgram Nova-2, and AssemblyAI go head-to-head across {len(meta['samples_tested'])} audio samples and {meta['total_api_calls']} API calls.</p>
  <div class="meta-line">{meta['date']} &middot; {meta['runs']} runs per sample &middot; {len(meta['providers_tested'])} providers</div>
</div>"""


def _leaderboard(summary, pnames, pcolors) -> str:
    # Sort by WER (primary metric)
    ranked = sorted(summary.items(), key=lambda x: x[1].get("avg_wer", 999))

    best_wer = min(s.get("avg_wer", 999) for s in summary.values())
    best_speed = min(s.get("avg_speed", 999) for s in summary.values())

    rows = ""
    for i, (pid, s) in enumerate(ranked):
        wer = s.get("avg_wer")
        speed = s.get("avg_speed")
        success = s.get("success_rate", 0)

        wer_str = f"{wer:.1%}" if wer is not None else "N/A"
        speed_str = f"{speed:.2f}s" if speed is not None else "N/A"

        wer_cls = " lb-best" if wer is not None and wer == best_wer else ""
        speed_cls = " lb-best" if speed is not None and speed == best_speed else ""

        rows += f"""<div class="lb-row">
  <span class="lb-rank">{i+1}</span>
  <span class="lb-name">{pnames.get(pid, pid)}</span>
  <span class="lb-val{wer_cls}">{wer_str}</span>
  <span class="lb-val{speed_cls}">{speed_str}</span>
  <span class="lb-val">{success:.0%}</span>
</div>"""

    return f"""
<h2>Results at a glance</h2>
<div class="leaderboard">
  <div class="lb-row lb-header">
    <span></span><span>Provider</span>
    <span style="text-align:right">WER</span>
    <span style="text-align:right">Speed</span>
    <span style="text-align:right">Uptime</span>
  </div>
  {rows}
</div>
<p class="prose" style="font-size:0.82rem">
  <strong>WER</strong> = word error rate (lower is better), duration-weighted across all samples with ground truth.<br>
  <strong>Speed</strong> = median API response time.<br>
  <strong>Uptime</strong> = percentage of successful API calls across all runs.
</p>"""


def _radar_section() -> str:
    return """
<h2>Overall comparison</h2>
<p class="prose">Normalized scores across five dimensions. Higher is better on all axes. Speed and accuracy inverted so that faster/more accurate = higher.</p>
<div class="chart-wrap" style="max-width:480px;margin:16px auto 32px">
  <canvas id="radarChart"></canvas>
</div>"""


def _wer_chart_section() -> str:
    return """
<div class="chart-wrap">
  <canvas id="werChart" height="260"></canvas>
</div>"""


def _noise_section(samples, pnames, pcolors) -> str:
    providers = list(next(iter(samples.values()))["providers"].keys()) if samples else []
    noise_data = {}
    for pid in providers:
        clean = samples.get("harvard_sentences", {}).get("providers", {}).get(pid, {}).get("wer")
        n15 = samples.get("noisy_15db", {}).get("providers", {}).get(pid, {}).get("wer")
        n5 = samples.get("noisy_5db", {}).get("providers", {}).get(pid, {}).get("wer")
        if clean is not None:
            noise_data[pid] = (clean, n15, n5)

    if not noise_data:
        return ""

    rows = ""
    for pid, (clean, n15, n5) in noise_data.items():
        deg = (n5 - clean) if n5 is not None and clean is not None else None
        deg_str = f'<span class="ng-arrow">+{deg:.0%}</span>' if deg is not None else ""
        rows += f"""<div class="ng-cell" style="font-weight:500">{pnames.get(pid,pid)}</div>
<div class="ng-cell">{clean:.1%}</div>
<div class="ng-cell">{n15:.1%}</div>
<div class="ng-cell">{n5:.1%} {deg_str}</div>"""

    return f"""
<h2>Noise robustness</h2>
<p class="prose">The same Harvard Sentences audio tested clean, with moderate background noise (15 dB SNR), and heavy noise (5 dB SNR). Pink noise was synthetically mixed at verified levels.</p>
<div class="noise-grid">
  <div class="ng-cell ng-header">Provider</div>
  <div class="ng-cell ng-header" style="text-align:right">Clean</div>
  <div class="ng-cell ng-header" style="text-align:right">15 dB</div>
  <div class="ng-cell ng-header" style="text-align:right">5 dB</div>
  {rows}
</div>"""


def _speed_section(samples, pnames, pcolors, providers, chart_data) -> str:
    return f"""
<h2>Speed</h2>
<p class="prose">Median API response time across 10 runs. Includes network latency. AssemblyAI uses an asynchronous API (upload + poll), inflating its wall-clock time relative to synchronous providers.</p>
<div class="chart-wrap" style="padding-left:12px">
  <canvas id="speedChart" height="260"></canvas>
</div>"""


def _accuracy_section(samples, pnames, pcolors, providers) -> str:
    rows = ""
    for sid, sdata in samples.items():
        ref_type = sdata.get("reference_type", "none")
        if ref_type == "none":
            continue
        ref_label = "GT" if ref_type == "ground_truth" else "CON"
        rows += "<tr>"
        rows += f'<td>{sdata["name"]} <span style="color:var(--text-3);font-size:0.7rem">{ref_label}</span></td>'
        for pid in providers:
            pdata = sdata["providers"].get(pid)
            if pdata and pdata.get("wer") is not None:
                wer = pdata["wer"]
                cls = "wer-good" if wer < 0.05 else "wer-mid" if wer < 0.15 else "wer-bad"
                rows += f'<td class="num {cls}">{wer:.1%}</td>'
            else:
                rows += '<td class="num" style="color:var(--text-3)">--</td>'
        rows += "</tr>\n"

    return f"""
<h2>Accuracy by sample</h2>
<p class="prose">Word error rate per audio sample. GT = verified ground truth transcript. CON = consensus reference (majority vote across all providers). Lower is better.</p>
<div style="overflow-x:auto">
<table class="data-table">
  <thead><tr>
    <th>Sample</th>
    {''.join(f'<th class="num">{pnames.get(pid,pid)}</th>' for pid in providers)}
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


def _cost_section(pnames, pcolors, providers) -> str:
    rows = ""
    for hours in PROJECTION_HOURS:
        rows += "<tr>"
        rows += f'<td>{hours:,}h</td>'
        for pid in providers:
            p = next((pp for pp in PROVIDERS if pp.id == pid), None)
            if p and p.price_per_minute > 0:
                cost = p.price_per_minute * 60 * hours
                rows += f'<td class="num">${cost:,.0f}</td>'
            else:
                rows += '<td class="num" style="color:var(--text-3)">TBD</td>'
        rows += "</tr>\n"

    per_min_row = ""
    for pid in providers:
        p = next((pp for pp in PROVIDERS if pp.id == pid), None)
        if p and p.price_per_minute > 0:
            per_min_row += f'<td class="num" style="font-weight:600">${p.price_per_minute:.4f}</td>'
        else:
            per_min_row += '<td class="num" style="color:var(--text-3)">TBD</td>'

    return f"""
<h2>Pricing</h2>
<p class="prose">Published per-minute rates projected across common workloads. Cohere pricing not announced at time of testing.</p>
<table class="data-table">
  <thead><tr>
    <th>Volume</th>
    {''.join(f'<th class="num">{pnames.get(pid,pid)}</th>' for pid in providers)}
  </tr></thead>
  <tbody>
    <tr><td>Per minute</td>{per_min_row}</tr>
    {rows}
  </tbody>
</table>"""


def _transcript_section(samples, pnames, pcolors, providers) -> str:
    tabs = ""
    panels = ""

    for i, (sid, sdata) in enumerate(samples.items()):
        active = "active" if i == 0 else ""
        tabs += f'<button class="t-tab {active}" onclick="showTab(\'{sid}\')" data-tab="{sid}">{sdata["name"]}</button>\n'

        content = ""

        if sdata.get("reference"):
            content += f'<div class="t-label">Reference ({sdata.get("reference_type", "?")})</div>\n'
            content += f'<div class="t-box" style="opacity:0.6">{sdata["reference"]}</div>\n'

        for pid in providers:
            pdata = sdata["providers"].get(pid)
            if not pdata:
                continue
            wer_note = f" &middot; {pdata['wer']:.1%} WER" if pdata.get("wer") is not None else ""
            content += f'<div class="t-label">{pnames.get(pid, pid)}{wer_note}</div>\n'

            if pdata.get("word_diff"):
                diff_html = " ".join(
                    f'<span class="word-{wd["status"]}">{wd["word"]}</span>'
                    for wd in pdata["word_diff"]
                )
                content += f'<div class="t-box">{diff_html}</div>\n'
            elif pdata.get("transcript"):
                content += f'<div class="t-box">{pdata["transcript"]}</div>\n'

        panels += f'<div class="t-panel {active}" data-panel="{sid}">{content}</div>\n'

    return f"""
<h2>Transcripts</h2>
<p class="prose">
  Side-by-side output with inline diff highlighting.
  <span class="word-match">correct</span> &middot;
  <span class="word-error">substitution</span> &middot;
  <span class="word-extra">insertion</span> &middot;
  <span class="word-missing">deletion</span>
</p>
<div class="transcript-tabs">{tabs}</div>
{panels}"""


def _methodology(meta, samples) -> str:
    sample_rows = ""
    for sid, sdata in samples.items():
        sample_rows += f'<tr><td>{sdata["name"]}</td><td>{sdata["category"]}</td><td class="num">{sdata.get("duration_seconds","?")}s</td><td>{sdata.get("reference_type","?")}</td><td style="font-size:0.78rem">{sdata["source"]}</td></tr>\n'

    return f"""
<details>
  <summary>Methodology</summary>
  <div class="inner">
    <h3>Configuration</h3>
    <p class="prose">{meta['runs']} runs per sample per provider. {meta['total_api_calls']} total API calls. WER computed with jiwer (Levenshtein word edit distance). Timing = wall-clock including network. RTFx = audio_duration / elapsed. Duration-weighted WER gives longer samples proportionally more influence on averages.</p>
    <h3>Samples</h3>
    <table class="data-table">
      <thead><tr><th>Name</th><th>Category</th><th class="num">Duration</th><th>Reference</th><th>Source</th></tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
    <h3>Limitations</h3>
    <p class="prose">API timing includes network latency, not pure inference speed. AssemblyAI's async API (upload + poll) inflates its wall-clock time. Results from a single test location. Consensus references bias toward the majority transcript. Accent samples may include brief unscripted speech after the reading passage, inflating WER for all providers equally.</p>
  </div>
</details>"""


# ── Chart Data ──────────────────────────────────────────────────────────────

def _build_chart_data(results: dict) -> dict:
    summary = results["summary"]
    samples = results["samples"]
    providers = results["metadata"]["providers_tested"]

    # Muted editorial palette
    colors = {
        "cohere": "#1d4ed8",
        "openai": "#059669",
        "deepgram": "#7c3aed",
        "assemblyai": "#c2410c",
    }

    # ── Radar chart ──
    radar = {
        "labels": ["Accuracy", "Speed", "Noise robustness", "Punctuation", "Reliability"],
        "datasets": [],
    }
    for pid in providers:
        s = summary.get(pid, {})
        p = next((pp for pp in PROVIDERS if pp.id == pid), None)

        accuracy = max(0, (1 - (s.get("avg_wer") or 0.5)) * 100)
        speed_score = max(0, 100 - ((s.get("avg_speed") or 10) * 8))
        punct = s.get("avg_punctuation", 0) * 100

        # Noise robustness: lower degradation = higher score
        clean_wer = samples.get("harvard_sentences", {}).get("providers", {}).get(pid, {}).get("wer")
        noisy_wer = samples.get("noisy_5db", {}).get("providers", {}).get(pid, {}).get("wer")
        if clean_wer is not None and noisy_wer is not None:
            degradation = noisy_wer - clean_wer
            noise_score = max(0, 100 - degradation * 500)
        else:
            noise_score = 50

        reliability = s.get("success_rate", 0) * 100

        c = colors.get(pid, "#888")
        radar["datasets"].append({
            "label": p.name if p else pid,
            "data": [round(accuracy, 1), round(speed_score, 1), round(noise_score, 1), round(punct, 1), round(reliability, 1)],
            "borderColor": c,
            "backgroundColor": c + "18",
            "pointBackgroundColor": c,
            "borderWidth": 2,
            "pointRadius": 3,
        })

    # ── Speed bar chart ──
    speed = {"labels": [], "datasets": []}
    sample_names = [samples[sid]["name"] for sid in samples]
    speed["labels"] = sample_names
    for pid in providers:
        p = next((pp for pp in PROVIDERS if pp.id == pid), None)
        data = []
        for sid in samples:
            pdata = samples[sid]["providers"].get(pid)
            if pdata and pdata.get("timing"):
                data.append(round(pdata["timing"]["median"], 2))
            else:
                data.append(0)
        speed["datasets"].append({
            "label": p.name if p else pid,
            "data": data,
            "backgroundColor": colors.get(pid, "#666") + "CC",
            "borderColor": colors.get(pid, "#666"),
            "borderWidth": 1,
            "borderRadius": 2,
        })

    # ── WER bar chart ──
    wer_chart = {"labels": [], "datasets": []}
    wer_samples = [sid for sid in samples if samples[sid].get("reference_type") != "none"]
    wer_chart["labels"] = [samples[sid]["name"] for sid in wer_samples]
    for pid in providers:
        p = next((pp for pp in PROVIDERS if pp.id == pid), None)
        data = []
        for sid in wer_samples:
            pdata = samples[sid]["providers"].get(pid)
            if pdata and pdata.get("wer") is not None:
                data.append(round(pdata["wer"] * 100, 1))
            else:
                data.append(0)
        wer_chart["datasets"].append({
            "label": p.name if p else pid,
            "data": data,
            "backgroundColor": colors.get(pid, "#666") + "CC",
            "borderColor": colors.get(pid, "#666"),
            "borderWidth": 1,
            "borderRadius": 2,
        })

    return {"radar": radar, "speed": speed, "wer": wer_chart}


def _chart_js() -> str:
    return """
function showTab(id) {
  document.querySelectorAll('.t-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.t-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.t-tab[data-tab="${id}"]`)?.classList.add('active');
  document.querySelector(`.t-panel[data-panel="${id}"]`)?.classList.add('active');
}

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
Chart.defaults.color = isDark ? '#a3a3a3' : '#4b5563';
Chart.defaults.borderColor = isDark ? '#2a2a2a' : '#e5e7eb';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

const legendOpts = { position: 'bottom', labels: { padding: 12, boxWidth: 10, usePointStyle: true, pointStyle: 'rect', font: { size: 10 } } };
const gridLight = { color: isDark ? '#1a1a1a' : '#f3f4f6' };

// Radar chart
const radarCtx = document.getElementById('radarChart');
if (radarCtx) {
  new Chart(radarCtx, {
    type: 'radar',
    data: CHART_DATA.radar,
    options: {
      responsive: true,
      scales: {
        r: {
          beginAtZero: true, max: 100,
          ticks: { stepSize: 25, display: false },
          grid: { color: isDark ? '#222' : '#e5e7eb' },
          angleLines: { color: isDark ? '#222' : '#e5e7eb' },
          pointLabels: { font: { size: 11, weight: '500' }, color: isDark ? '#ccc' : '#374151' },
        }
      },
      plugins: { legend: legendOpts },
    }
  });
}

// Speed chart
const speedCtx = document.getElementById('speedChart');
if (speedCtx) {
  new Chart(speedCtx, {
    type: 'bar',
    data: CHART_DATA.speed,
    options: {
      responsive: true,
      indexAxis: 'y',
      scales: {
        x: { title: { display: true, text: 'Median response time (seconds)', font: { size: 11 } }, grid: gridLight },
        y: { grid: { display: false }, afterFit(axis) { axis.width = 180; } },
      },
      layout: { padding: { left: 0 } },
      plugins: { legend: legendOpts },
      barPercentage: 0.7, categoryPercentage: 0.85,
    }
  });
}

// WER chart (horizontal)
const werCtx = document.getElementById('werChart');
if (werCtx) {
  new Chart(werCtx, {
    type: 'bar',
    data: CHART_DATA.wer,
    options: {
      responsive: true,
      indexAxis: 'y',
      scales: {
        x: { title: { display: true, text: 'Word error rate (%)', font: { size: 11 } }, beginAtZero: true, grid: gridLight },
        y: { grid: { display: false }, afterFit(axis) { axis.width = 180; } },
      },
      layout: { padding: { left: 0 } },
      plugins: { legend: legendOpts },
      barPercentage: 0.7, categoryPercentage: 0.85,
    }
  });
}
"""
