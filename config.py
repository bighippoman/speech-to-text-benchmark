"""
Configuration for ASR Model Shootout.
Defines audio samples, provider metadata, and pricing.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# ── Directory paths ─────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_CACHE_DIR = os.path.join(PROJECT_DIR, "audio_cache")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

# ── Audio sample definitions ────────────────────────────────────────────────


@dataclass
class AudioSample:
    id: str
    name: str
    category: str  # Baseline, Noise, Accent, Language, Speed, Multi-speaker
    url: Optional[str]  # Download URL (None = generated/synthetic)
    reference: Optional[str]  # Ground truth transcript (None = consensus)
    language: str  # ISO 639-1
    description: str
    source: str  # Where it's from (for methodology section)
    trim_seconds: Optional[float] = None  # Trim to this duration after download
    trim_start: Optional[float] = None  # Skip this many seconds from start
    synthetic_from: Optional[str] = None  # Generate from this sample ID
    snr_db: Optional[float] = None  # Target SNR for noise mixing
    frozen_reference_from: Optional[str] = None  # Use this sample's consensus as frozen ref


# ── Reference transcripts ───────────────────────────────────────────────────

# Harvard Sentences List 1 (OSR female American reader)
HARVARD_SENTENCES_L1 = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue "
    "background. It's easy to tell the depth of a well. These days a chicken leg "
    "is a rare dish. Rice is often served in round bowls. The juice of lemons "
    "makes fine punch. The box was thrown beside the parked truck. The hogs were "
    "fed chopped corn and garbage. Four hours of steady work faced us. A large "
    "size in stockings is hard to sell."
)

# Rainbow Passage — the ACTUAL passage used in IDEA dialect archive recordings
# FULL version (some speakers read all of it, some stop after "finds it")
RAINBOW_PASSAGE_FULL = (
    "When the sunlight strikes raindrops in the air, they act as a prism and "
    "form a rainbow. The rainbow is a division of white light into many beautiful "
    "colors. These take the shape of a long round arch with its path high above "
    "and its two ends apparently beyond the horizon. There is, according to "
    "legend, a boiling pot of gold at one end. People look, but no one ever "
    "finds it. When a man looks for something beyond his reach, his friends say "
    "he is looking for the pot of gold at the end of the rainbow."
)

# SHORT version — some speakers only read through the first paragraph
RAINBOW_PASSAGE_SHORT = (
    "When the sunlight strikes raindrops in the air, they act as a prism and "
    "form a rainbow. The rainbow is a division of white light into many beautiful "
    "colors. These take the shape of a long round arch with its path high above "
    "and its two ends apparently beyond the horizon. There is, according to "
    "legend, a boiling pot of gold at one end. People look, but no one ever "
    "finds it."
)

# Gettysburg Address — hardcoded ground truth (public domain)
GETTYSBURG_ADDRESS = (
    "Four score and seven years ago our fathers brought forth on this continent "
    "a new nation, conceived in liberty, and dedicated to the proposition that "
    "all men are created equal. Now we are engaged in a great civil war, testing "
    "whether that nation, or any nation so conceived and so dedicated, can long "
    "endure. We are met on a great battlefield of that war. We have come to "
    "dedicate a portion of that field, as a final resting place for those who "
    "here gave their lives that that nation might live. It is altogether fitting "
    "and proper that we should do this. But, in a larger sense, we cannot "
    "dedicate, we cannot consecrate, we cannot hallow this ground. The brave "
    "men, living and dead, who struggled here, have consecrated it, far above "
    "our poor power to add or detract. The world will little note nor long "
    "remember what we say here, but it can never forget what they did here. "
    "It is for us the living, rather, to be dedicated here to the unfinished "
    "work which they who fought here have thus far so nobly advanced. It is "
    "rather for us to be here dedicated to the great task remaining before us, "
    "that from these honored dead we take increased devotion to that cause for "
    "which they gave the last full measure of devotion, that we here highly "
    "resolve that these dead shall not have died in vain, that this nation, "
    "under God, shall have a new birth of freedom, and that government of the "
    "people, by the people, for the people, shall not perish from the earth."
)


SAMPLES: list[AudioSample] = [
    # ── Baseline ────────────────────────────────────────────────────────
    AudioSample(
        id="clean_medium",
        name="Clean English (10s)",
        category="Baseline",
        url="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
        reference=None,  # consensus on first run, then frozen for noise comparison
        language="en",
        description="LibriSpeech-style clean read speech, single speaker, ~10s",
        source="HuggingFace Narsil/asr_dummy (LibriSpeech derivative)",
    ),
    AudioSample(
        id="harvard_sentences",
        name="Harvard Sentences (33s)",
        category="Baseline",
        url="http://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
        reference=HARVARD_SENTENCES_L1,
        language="en",
        description="10 Harvard sentences, female American, 8kHz telephone-band, ~33s",
        source="Open Speech Repository (public domain)",
    ),
    # ── Noise (source = harvard_sentences which has ground truth) ──────
    AudioSample(
        id="noisy_15db",
        name="Noisy (15 dB SNR)",
        category="Noise",
        url=None,
        reference=HARVARD_SENTENCES_L1,  # FIXED: use same ground truth as source
        language="en",
        description="Harvard Sentences + pink noise at 15 dB SNR (moderate background)",
        source="Synthetic — OSR + generated pink noise (verified 15.0 dB)",
        synthetic_from="harvard_sentences",
        snr_db=15.0,
    ),
    AudioSample(
        id="noisy_5db",
        name="Noisy (5 dB SNR)",
        category="Noise",
        url=None,
        reference=HARVARD_SENTENCES_L1,  # FIXED: use same ground truth as source
        language="en",
        description="Harvard Sentences + pink noise at 5 dB SNR (heavy background)",
        source="Synthetic — OSR + generated pink noise (verified 5.0 dB)",
        synthetic_from="harvard_sentences",
        snr_db=5.0,
    ),
    # ── Accent (FIXED: correct passage + trim to skip preamble) ────────
    AudioSample(
        id="accent_indian",
        name="Indian-Accented English",
        category="Accent",
        url="https://www.dialectsarchive.com/wp-content/uploads/2011/08/india1.mp3",
        reference=RAINBOW_PASSAGE_SHORT,
        language="en",
        description="Native Hindi speaker reading short Rainbow Passage (preamble trimmed, unscripted speech excluded)",
        source="IDEA International Dialects of English Archive",
        trim_start=6.0,   # Skip copyright preamble
        trim_seconds=24.5,  # Short passage ends at "finds it" — 25.5s still catches "I." from AssemblyAI
    ),
    AudioSample(
        id="accent_french",
        name="French-Accented English",
        category="Accent",
        url=None,  # Manually trimmed by editor — already in audio_cache
        reference=RAINBOW_PASSAGE_FULL,
        language="en",
        description="Native French speaker reading full Rainbow Passage (manually trimmed to passage only)",
        source="IDEA International Dialects of English Archive",
    ),
    # ── Long-form (FIXED: hardcoded reference, trim LibriVox preamble) ─
    AudioSample(
        id="gettysburg",
        name="Gettysburg Address",
        category="Long-form",
        url=None,  # Manually trimmed by editor — already in audio_cache
        reference=GETTYSBURG_ADDRESS,
        language="en",
        description="Full Gettysburg Address, LibriVox narration, manually trimmed to start at 'Four score'",
        source="LibriVox / Internet Archive (public domain)",
    ),
    # ── Historical / Archival ───────────────────────────────────────────
    AudioSample(
        id="mlk_speech",
        name="MLK Speech (13s)",
        category="Historical",
        url="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        reference=None,  # consensus — archival clip with no published transcript
        language="en",
        description="Martin Luther King Jr. excerpt, archival audio quality, ~13s",
        source="HuggingFace Narsil/asr_dummy (public domain speech)",
    ),
]


# ── Provider metadata ───────────────────────────────────────────────────────


@dataclass
class ProviderConfig:
    id: str
    name: str
    color: str  # Hex color for charts
    env_key: str  # Environment variable name for API key
    price_per_minute: float  # USD per minute of audio
    model_name: str
    notes: str = ""


PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        id="cohere",
        name="Cohere Transcribe",
        color="#FF6B6B",
        env_key="CO_API_KEY",
        price_per_minute=0.0,  # Not yet announced at time of writing
        model_name="cohere-transcribe-03-2026",
        notes="Price TBD — currently in preview/launch period",
    ),
    ProviderConfig(
        id="openai",
        name="OpenAI Whisper",
        color="#10A37F",
        env_key="OPENAI_API_KEY",
        price_per_minute=0.006,
        model_name="whisper-1",
        notes="$0.006/min — battle-tested, widely adopted",
    ),
    ProviderConfig(
        id="deepgram",
        name="Deepgram Nova-2",
        color="#13EF93",
        env_key="DEEPGRAM_API_KEY",
        price_per_minute=0.0043,
        model_name="nova-2",
        notes="$0.0043/min — fastest API response times",
    ),
    ProviderConfig(
        id="assemblyai",
        name="AssemblyAI",
        color="#4C6EF5",
        env_key="ASSEMBLYAI_API_KEY",
        price_per_minute=0.015,
        model_name="best",
        notes="$0.015/min (Async model) — strong accuracy",
    ),
]


# ── Pricing projections ────────────────────────────────────────────────────

PROJECTION_HOURS = [1, 10, 100, 1000]  # Hours of audio for cost table


def get_provider_by_id(provider_id: str) -> Optional[ProviderConfig]:
    for p in PROVIDERS:
        if p.id == provider_id:
            return p
    return None


def get_sample_by_id(sample_id: str) -> Optional[AudioSample]:
    for s in SAMPLES:
        if s.id == sample_id:
            return s
    return None
