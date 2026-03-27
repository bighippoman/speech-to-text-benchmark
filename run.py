#!/usr/bin/env python3
"""
ASR Model Shootout — Comprehensive benchmark runner.
Tests Cohere Transcribe vs OpenAI Whisper vs Deepgram Nova-2 vs AssemblyAI.

Usage:
  python run.py                         # Full run (10 iterations, all samples)
  python run.py --runs 3                # Quick test with 3 iterations
  python run.py --providers cohere openai  # Test specific providers
  python run.py --dry-run               # Validate setup without API calls
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

from config import SAMPLES, PROVIDERS, AUDIO_CACHE_DIR, REPORTS_DIR
from providers import transcribe, TranscriptionResult
from samples import prepare_all_samples, get_audio_duration
from analysis import (
    compute_wer, compute_wer_details, score_punctuation, score_capitalization,
    compute_stats, word_overlap, compute_consensus_reference, compute_word_diff,
    compute_rtfx,
)
from report import generate_report


# ── Argument Parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR Model Shootout — benchmark runner for AINH article"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of runs per sample per provider (default: 10)"
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Provider IDs to test (default: all). Options: cohere openai deepgram assemblyai"
    )
    parser.add_argument(
        "--samples", nargs="+", default=None,
        help="Sample IDs to test (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML report path (default: reports/shootout_TIMESTAMP.html)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate setup: check API keys, download samples, skip API calls"
    )
    return parser.parse_args()


# ── Setup Validation ────────────────────────────────────────────────────────

def check_api_keys(provider_ids: list[str]) -> dict[str, str]:
    """Check for API keys in environment variables. Returns {provider_id: key}."""
    keys = {}
    print("\n  API Keys:")
    for pid in provider_ids:
        p = None
        for pp in PROVIDERS:
            if pp.id == pid:
                p = pp
                break
        if not p:
            print(f"    {pid}: UNKNOWN PROVIDER")
            continue
        key = os.environ.get(p.env_key)
        if key:
            masked = key[:6] + "..." + key[-4:]
            print(f"    {p.name}: {masked}")
            keys[pid] = key
        else:
            print(f"    {p.name}: MISSING ({p.env_key} not set)")
    return keys


def estimate_cost_and_time(n_samples: int, n_providers: int, n_runs: int):
    """Print estimated cost and runtime."""
    total_calls = n_samples * n_providers * n_runs
    # Rough estimates: ~3s per call average, AssemblyAI slower
    est_time_min = total_calls * 4 / 60  # 4s average including AssemblyAI
    est_cost = total_calls * 0.002  # Very rough average

    print(f"\n  Estimated:")
    print(f"    API calls:  {total_calls}")
    print(f"    Runtime:    ~{est_time_min:.0f} minutes")
    print(f"    API cost:   ~${est_cost:.2f} (varies by provider)")


# ── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(
    sample_paths: dict[str, str],
    provider_keys: dict[str, str],
    n_runs: int,
    sample_filter: list[str] | None = None,
) -> dict:
    """
    Run the full benchmark suite.
    Returns the results dict for report generation.
    """
    # Filter samples
    active_samples = [s for s in SAMPLES if s.id in sample_paths]
    if sample_filter:
        active_samples = [s for s in active_samples if s.id in sample_filter]

    active_providers = list(provider_keys.keys())

    total_calls = len(active_samples) * len(active_providers) * n_runs
    completed = 0

    print(f"\n{'='*60}")
    print(f"RUNNING BENCHMARK")
    print(f"{'='*60}")
    print(f"  {len(active_samples)} samples x {len(active_providers)} providers x {n_runs} runs = {total_calls} calls")

    # Results storage
    # raw_results[sample_id][provider_id] = list of TranscriptionResult
    raw_results: dict[str, dict[str, list[TranscriptionResult]]] = {}

    start_time = time.time()

    for sample in active_samples:
        filepath = sample_paths[sample.id]
        raw_results[sample.id] = {}

        print(f"\n  [{sample.category}] {sample.name}")

        for pid in active_providers:
            api_key = provider_keys[pid]
            pname = pid
            for pp in PROVIDERS:
                if pp.id == pid:
                    pname = pp.name
                    break

            raw_results[sample.id][pid] = []

            for run_idx in range(n_runs):
                completed += 1
                pct = completed / total_calls * 100
                elapsed_total = time.time() - start_time
                eta = (elapsed_total / completed * (total_calls - completed)) if completed > 0 else 0

                print(
                    f"\r    {pname}: run {run_idx+1}/{n_runs}  "
                    f"[{pct:.0f}% | ETA {eta/60:.0f}m]   ",
                    end="", flush=True
                )

                result = transcribe(pid, filepath, sample.language, api_key)
                raw_results[sample.id][pid].append(result)

                if result.error:
                    print(f"\n    ERROR: {result.error[:80]}")

                # Small delay to be respectful to APIs
                if run_idx < n_runs - 1:
                    time.sleep(0.3)

            # Print summary for this provider+sample
            successes = [r for r in raw_results[sample.id][pid] if r.text is not None]
            times = [r.elapsed for r in successes]
            if times:
                med = sorted(times)[len(times) // 2]
                print(f"\r    {pname}: {len(successes)}/{n_runs} ok, median {med:.2f}s          ")
            else:
                print(f"\r    {pname}: ALL FAILED                    ")

    total_elapsed = time.time() - start_time
    print(f"\n  Benchmark completed in {total_elapsed/60:.1f} minutes")

    # ── Build results dict ──────────────────────────────────────────────

    return _build_results(raw_results, sample_paths, active_samples, active_providers, n_runs)


def _build_results(
    raw_results: dict,
    sample_paths: dict[str, str],
    active_samples: list,
    active_providers: list[str],
    n_runs: int,
) -> dict:
    """Transform raw benchmark data into the report-ready structure."""

    results = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "runs": n_runs,
            "samples_tested": [s.id for s in active_samples],
            "providers_tested": active_providers,
            "total_api_calls": len(active_samples) * len(active_providers) * n_runs,
        },
        "samples": {},
        "summary": {},
    }

    # Per-provider aggregate stats
    provider_agg: dict[str, dict] = {pid: {
        "wers": [], "wer_durations": [],  # parallel lists for duration-weighted avg
        "speeds": [], "puncts": [], "caps": [], "rtfxs": [],
        "total_runs": 0, "failed_runs": 0,
    } for pid in active_providers}

    for sample in active_samples:
        sid = sample.id
        filepath = sample_paths[sid]

        try:
            duration = get_audio_duration(filepath)
        except Exception:
            duration = 0

        # Determine reference text
        reference = sample.reference
        reference_type = "ground_truth" if reference else "none"

        # For samples without ground truth, compute consensus reference
        if not reference:
            consensus_candidates = []
            for pid in active_providers:
                runs = raw_results.get(sid, {}).get(pid, [])
                successes = [r for r in runs if r.text]
                if successes:
                    # Use the median-length transcript as representative
                    sorted_by_len = sorted(successes, key=lambda r: len(r.text))
                    consensus_candidates.append(sorted_by_len[len(sorted_by_len) // 2].text)
            if consensus_candidates:
                reference = compute_consensus_reference(consensus_candidates)
                reference_type = "consensus"

        sample_data = {
            "name": sample.name,
            "category": sample.category,
            "description": sample.description,
            "source": sample.source,
            "duration_seconds": round(duration, 1) if duration else "N/A",
            "reference": reference,
            "reference_type": reference_type,
            "providers": {},
        }

        for pid in active_providers:
            runs = raw_results.get(sid, {}).get(pid, [])
            successes = [r for r in runs if r.text is not None]
            failures = [r for r in runs if r.text is None]

            provider_agg[pid]["total_runs"] += len(runs)
            provider_agg[pid]["failed_runs"] += len(failures)

            if not successes:
                sample_data["providers"][pid] = {
                    "name": _pname(pid),
                    "color": _pcolor(pid),
                    "timing": None,
                    "wer": None,
                    "punctuation": None,
                    "capitalization": None,
                    "rtfx": None,
                    "transcript": None,
                    "word_diff": None,
                    "error": failures[0].error if failures else "All runs failed",
                }
                continue

            # Timing stats
            times = [r.elapsed for r in successes]
            timing_stats = compute_stats(times)

            # Pick the median-time transcript as representative
            sorted_runs = sorted(successes, key=lambda r: r.elapsed)
            median_run = sorted_runs[len(sorted_runs) // 2]
            transcript = median_run.text

            # WER
            wer = None
            wer_details = None
            if reference:
                wer = compute_wer(reference, transcript)
                wer_details = compute_wer_details(reference, transcript)
                provider_agg[pid]["wers"].append(wer)
                provider_agg[pid]["wer_durations"].append(duration if duration else 1.0)

            # Quality scores
            punct = score_punctuation(transcript)
            cap = score_capitalization(transcript)
            provider_agg[pid]["puncts"].append(punct["score"])
            provider_agg[pid]["caps"].append(cap["score"])
            provider_agg[pid]["speeds"].append(timing_stats.median)

            # RTFx
            rtfx_values = []
            if duration > 0:
                rtfx_values = [compute_rtfx(duration, r.elapsed) for r in successes]
                provider_agg[pid]["rtfxs"].extend(rtfx_values)
            rtfx_stats = compute_stats(rtfx_values) if rtfx_values else None

            # Word diff
            diff = None
            if reference:
                diff = compute_word_diff(reference, transcript)

            sample_data["providers"][pid] = {
                "name": _pname(pid),
                "color": _pcolor(pid),
                "timing": {
                    "median": timing_stats.median,
                    "mean": timing_stats.mean,
                    "std": timing_stats.std,
                    "min_val": timing_stats.min_val,
                    "max_val": timing_stats.max_val,
                    "p25": timing_stats.p25,
                    "p75": timing_stats.p75,
                    "values": timing_stats.values,
                },
                "wer": wer,
                "wer_details": wer_details,
                "punctuation": punct,
                "capitalization": cap,
                "rtfx": {
                    "median": rtfx_stats.median if rtfx_stats else 0,
                    "mean": rtfx_stats.mean if rtfx_stats else 0,
                    "std": rtfx_stats.std if rtfx_stats else 0,
                } if rtfx_stats else None,
                "transcript": transcript,
                "word_diff": diff,
                "error": None,
            }

        results["samples"][sid] = sample_data

    # Build summary
    for pid in active_providers:
        agg = provider_agg[pid]
        import numpy as np

        # Duration-weighted WER: longer samples count more
        if agg["wers"] and agg["wer_durations"]:
            wers = np.array(agg["wers"])
            durs = np.array(agg["wer_durations"])
            weighted_wer = float(np.average(wers, weights=durs))
            unweighted_wer = float(np.mean(wers))
        else:
            weighted_wer = None
            unweighted_wer = None

        results["summary"][pid] = {
            "avg_wer": weighted_wer,  # duration-weighted (primary metric)
            "avg_wer_unweighted": unweighted_wer,  # simple mean (for comparison)
            "avg_speed": float(np.median(agg["speeds"])) if agg["speeds"] else None,
            "avg_punctuation": float(np.mean(agg["puncts"])) if agg["puncts"] else 0,
            "avg_capitalization": float(np.mean(agg["caps"])) if agg["caps"] else 0,
            "avg_rtfx": float(np.mean(agg["rtfxs"])) if agg["rtfxs"] else 0,
            "success_rate": (
                (agg["total_runs"] - agg["failed_runs"]) / agg["total_runs"]
                if agg["total_runs"] > 0 else 0
            ),
            "total_runs": agg["total_runs"],
            "failed_runs": agg["failed_runs"],
        }

    return results


def _pname(pid):
    for p in PROVIDERS:
        if p.id == pid:
            return p.name
    return pid


def _pcolor(pid):
    for p in PROVIDERS:
        if p.id == pid:
            return p.color
    return "#888"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  ASR MODEL SHOOTOUT")
    print("  Cohere  |  Whisper  |  Deepgram  |  AssemblyAI")
    print("=" * 60)

    # Determine which providers to test
    provider_ids = args.providers or [p.id for p in PROVIDERS]
    provider_ids = [pid for pid in provider_ids if any(p.id == pid for p in PROVIDERS)]

    if not provider_ids:
        print("ERROR: No valid providers specified.")
        sys.exit(1)

    # Check API keys
    keys = check_api_keys(provider_ids)
    active_providers = [pid for pid in provider_ids if pid in keys]

    if not active_providers:
        print("\nERROR: No API keys found. Set environment variables:")
        for p in PROVIDERS:
            if p.id in provider_ids:
                print(f"  export {p.env_key}=your_key_here")
        sys.exit(1)

    print(f"\n  Active providers: {', '.join(active_providers)}")

    # Download and prepare audio samples
    sample_paths = prepare_all_samples(
        [s for s in SAMPLES if not args.samples or s.id in args.samples]
    )

    if not sample_paths:
        print("\nERROR: No audio samples available. Check your internet connection.")
        sys.exit(1)

    # Estimate time/cost
    estimate_cost_and_time(len(sample_paths), len(active_providers), args.runs)

    if args.dry_run:
        print("\n  DRY RUN complete — setup validated, no API calls made.")
        print(f"  Ready to test {len(sample_paths)} samples x {len(active_providers)} providers x {args.runs} runs")
        return

    # Run benchmark
    results = run_benchmark(
        sample_paths=sample_paths,
        provider_keys=keys,
        n_runs=args.runs,
        sample_filter=args.samples,
    )

    # Save raw results as JSON
    json_path = os.path.join(REPORTS_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Raw results: {json_path}")

    # Generate HTML report
    output = args.output or os.path.join(
        REPORTS_DIR, f"shootout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    generate_report(results, output)

    # Print quick summary
    print(f"\n{'='*60}")
    print("QUICK SUMMARY")
    print(f"{'='*60}")
    summary = results["summary"]
    for pid in active_providers:
        s = summary.get(pid, {})
        name = _pname(pid)
        wer_str = f"{s['avg_wer']:.1%}" if s.get("avg_wer") is not None else "N/A"
        speed_str = f"{s['avg_speed']:.2f}s" if s.get("avg_speed") is not None else "N/A"
        print(f"  {name:<25} WER: {wer_str:<10} Speed: {speed_str:<10} Success: {s.get('success_rate',0):.0%}")

    print(f"\n  Open the report: file://{os.path.abspath(output)}")
    print("  Done!")


if __name__ == "__main__":
    main()
