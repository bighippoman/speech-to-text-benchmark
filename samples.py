"""
Audio sample management: download, cache, trim, and noise mixing.
"""

import os
import requests
import numpy as np
from typing import Optional

from config import AudioSample, AUDIO_CACHE_DIR, SAMPLES


def ensure_cache_dir():
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


def cached_path(sample: AudioSample) -> str:
    """Return the expected local filepath for a sample."""
    if sample.url:
        ext = os.path.splitext(sample.url.split("?")[0])[1] or ".wav"
    elif sample.synthetic_from:
        ext = ".wav"  # Synthetic samples are always WAV
    else:
        ext = ".wav"
    return os.path.join(AUDIO_CACHE_DIR, f"{sample.id}{ext}")


def download_file(url: str, dest: str, label: str = "") -> bool:
    """Download a file with progress indication."""
    if os.path.exists(dest):
        size = os.path.getsize(dest) / (1024 * 1024)
        print(f"    [cached] {label} ({size:.1f} MB)")
        return True
    print(f"    Downloading {label}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ASR-Benchmark/1.0"}
        r = requests.get(url, timeout=120, stream=True, headers=headers)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r    Downloading {label}... {pct:.0f}%", end="", flush=True)
        size = os.path.getsize(dest) / (1024 * 1024)
        print(f"\r    Downloaded {label} ({size:.1f} MB)          ")
        return True
    except Exception as e:
        print(f"    FAILED to download {label}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def trim_audio(filepath: str, max_seconds: float, output_path: str,
               start_seconds: float = 0.0) -> str:
    """Trim audio from start_seconds to start_seconds+max_seconds. Save as WAV."""
    import soundfile as sf

    data, sr = sf.read(filepath)
    start_sample = int(start_seconds * sr)
    if start_sample > 0 and start_sample < len(data):
        data = data[start_sample:]
        print(f"    Skipped first {start_seconds:.0f}s (preamble)")
    max_samples = int(max_seconds * sr)
    if len(data) > max_samples:
        data = data[:max_samples]
    duration = len(data) / sr
    print(f"    Trimmed to {duration:.1f}s ({len(data)} samples @ {sr}Hz)")
    sf.write(output_path, data, sr)
    return output_path


def get_audio_duration(filepath: str) -> float:
    """Get audio duration in seconds."""
    import soundfile as sf
    info = sf.info(filepath)
    return info.duration


def generate_pink_noise(num_samples: int) -> np.ndarray:
    """Generate pink (1/f) noise using the Voss-McCartney algorithm."""
    # Use 16 octaves for smooth pink noise
    num_octaves = 16
    pink = np.zeros(num_samples)
    for i in range(num_octaves):
        # Each octave has a lower update rate
        period = 2 ** i
        n_values = (num_samples + period - 1) // period
        values = np.random.randn(n_values)
        # Repeat each value for 'period' samples
        repeated = np.repeat(values, period)[:num_samples]
        pink += repeated
    # Normalize to unit variance
    pink = pink / np.std(pink)
    return pink


def mix_with_noise(filepath: str, snr_db: float, output_path: str) -> str:
    """
    Mix clean audio with pink noise at target SNR.
    Returns path to noisy WAV file.
    """
    import soundfile as sf

    data, sr = sf.read(filepath)

    # Handle stereo by converting to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Generate pink noise matching audio length
    noise = generate_pink_noise(len(data))

    # Calculate scaling factor for target SNR
    signal_power = np.mean(data ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0 or signal_power == 0:
        sf.write(output_path, data, sr)
        return output_path

    # SNR = 10 * log10(signal_power / (scale^2 * noise_power))
    # scale = sqrt(signal_power / (noise_power * 10^(SNR/10)))
    scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
    mixed = data + scale * noise

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    sf.write(output_path, mixed, sr)
    print(f"    Mixed noise at {snr_db} dB SNR → {output_path}")
    return output_path


def prepare_sample(sample: AudioSample, all_samples: dict[str, str]) -> Optional[str]:
    """
    Prepare a single audio sample. Returns filepath or None on failure.

    Args:
        sample: The sample definition
        all_samples: Dict of sample_id → filepath for already-prepared samples
    """
    ensure_cache_dir()
    dest = cached_path(sample)

    # Check for pre-existing manual files (e.g. editor-trimmed audio)
    if not sample.url and not sample.synthetic_from:
        # Look for any matching file in cache
        for ext in [".wav", ".flac", ".mp3", ".ogg"]:
            manual_path = os.path.join(AUDIO_CACHE_DIR, f"{sample.id}{ext}")
            if os.path.exists(manual_path):
                print(f"    [manual] {sample.name}")
                return manual_path
        print(f"    SKIP {sample.name}: no URL, not synthetic, no manual file in cache")
        return None

    # Synthetic samples (noise mixing)
    if sample.synthetic_from:
        source_path = all_samples.get(sample.synthetic_from)
        if not source_path:
            print(f"    SKIP {sample.name}: source sample '{sample.synthetic_from}' not available")
            return None
        if os.path.exists(dest):
            print(f"    [cached] {sample.name}")
            return dest
        return mix_with_noise(source_path, sample.snr_db, dest)

    # Downloadable samples
    if sample.url:
        raw_dest = os.path.join(AUDIO_CACHE_DIR, f"{sample.id}_raw{os.path.splitext(sample.url.split('?')[0])[1]}")
        success = download_file(sample.url, raw_dest, sample.name)
        if not success:
            return None

        # Trim if needed (supports both trim_start and trim_seconds)
        needs_trim = sample.trim_seconds or sample.trim_start
        if needs_trim:
            trimmed_dest = os.path.join(AUDIO_CACHE_DIR, f"{sample.id}.wav")
            if os.path.exists(trimmed_dest):
                print(f"    [cached] {sample.name} (trimmed)")
                return trimmed_dest
            try:
                max_sec = sample.trim_seconds or 9999.0
                start_sec = sample.trim_start or 0.0
                return trim_audio(raw_dest, max_sec, trimmed_dest, start_seconds=start_sec)
            except Exception as e:
                print(f"    FAILED to trim {sample.name}: {e}")
                return raw_dest  # Fall back to untrimmed
        return raw_dest

    print(f"    SKIP {sample.name}: no URL and not synthetic")
    return None


def prepare_all_samples(samples: Optional[list[AudioSample]] = None) -> dict[str, str]:
    """
    Download and prepare all audio samples.
    Returns dict of sample_id → filepath.
    """
    if samples is None:
        samples = SAMPLES

    print("\n" + "=" * 60)
    print("PREPARING AUDIO SAMPLES")
    print("=" * 60)

    prepared: dict[str, str] = {}

    # First pass: non-synthetic samples
    for sample in samples:
        if sample.synthetic_from:
            continue
        print(f"\n  [{sample.category}] {sample.name}")
        path = prepare_sample(sample, prepared)
        if path:
            prepared[sample.id] = path
            try:
                dur = get_audio_duration(path)
                print(f"    Duration: {dur:.1f}s")
            except Exception:
                pass

    # Second pass: synthetic samples (depend on first pass)
    for sample in samples:
        if not sample.synthetic_from:
            continue
        print(f"\n  [{sample.category}] {sample.name}")
        path = prepare_sample(sample, prepared)
        if path:
            prepared[sample.id] = path
            try:
                dur = get_audio_duration(path)
                print(f"    Duration: {dur:.1f}s")
            except Exception:
                pass

    print(f"\n  Prepared {len(prepared)}/{len(samples)} samples")
    return prepared
