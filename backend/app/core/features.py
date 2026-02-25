"""
Pure synchronous feature extraction — no async, no event loop issues.
Called from a ThreadPoolExecutor in tracks.py.

Strategy (Alt 2 — two-pass):
  Pass 1: Read full duration instantly via soundfile (no decode).
  Pass 2: Load up to ANALYSIS_DUR seconds at low SR for spectral/rhythm work.
  Structure and duration_sec always reflect the full file.
"""
import io
from typing import Any
import numpy as np
import structlog

log = structlog.get_logger()

N_MFCC       = 13
HOP_LEN      = 1024
N_FFT        = 2048
ANALYSIS_DUR = 60   # seconds of audio used for spectral/rhythm analysis
SR           = 16000

CAMELOT = {
    (0,True):"8B",(1,True):"3B",(2,True):"10B",(3,True):"5B",(4,True):"12B",(5,True):"7B",
    (6,True):"2B",(7,True):"9B",(8,True):"4B",(9,True):"11B",(10,True):"6B",(11,True):"1B",
    (0,False):"5A",(1,False):"12A",(2,False):"7A",(3,False):"2A",(4,False):"9A",(5,False):"4A",
    (6,False):"11A",(7,False):"6A",(8,False):"1A",(9,False):"8A",(10,False):"3A",(11,False):"10A",
}
KEYS    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NUMERAL = {0:"I",2:"ii",3:"bIII",4:"III",5:"IV",6:"bV",7:"V",8:"bVI",9:"vi",10:"bVII",11:"vii"}

COMMON_PROGRESSIONS = {
    "I-IV-V":    [0,5,7],
    "I-V-vi-IV": [0,7,9,5],
    "I-vi-IV-V": [0,9,5,7],
    "ii-V-I":    [2,7,0],
    "I-IV-vi-V": [0,5,9,7],
}


def _get_full_duration(raw_bytes: bytes) -> float:
    """
    Read the file's real duration without decoding audio.
    Falls back to None if soundfile can't handle the format.
    """
    try:
        import soundfile as sf
        buf = io.BytesIO(raw_bytes)
        info = sf.info(buf)
        return info.duration
    except Exception:
        return None


def _detect_key(chroma_mean):
    major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    mc = [float(np.corrcoef(np.roll(major,-i), chroma_mean)[0,1]) for i in range(12)]
    nc = [float(np.corrcoef(np.roll(minor,-i), chroma_mean)[0,1]) for i in range(12)]
    bm, bn = int(np.argmax(mc)), int(np.argmax(nc))
    if mc[bm] >= nc[bn]:
        return bm, True,  round(mc[bm], 3)
    return bn, False, round(nc[bn], 3)


def _chords(chroma_mean, key_idx):
    top       = np.argsort(chroma_mean)[::-1][:4]
    intervals = sorted(int(n - key_idx) % 12 for n in top)
    best, best_score = "I-IV-V", 0
    for name, pattern in COMMON_PROGRESSIONS.items():
        score = sum(1 for p in pattern if p in intervals)
        if score > best_score:
            best_score, best = score, name
    chords = [{"chord": KEYS[(key_idx+i)%12], "function": NUMERAL.get(i,"?")} for i in intervals]
    return best, " — ".join(NUMERAL.get(i,"?") for i in intervals), chords


def _structure(full_duration: float, bpm: float) -> list:
    """
    Build song structure based on the FULL duration of the file,
    not the analysis window. Timestamps are always correct.
    """
    bar = (60.0 / max(float(bpm), 60)) * 4
    bars = full_duration / bar

    if bars < 16:
        cuts   = [0, 0.2, 0.6, 1.0]
        labels = ["intro", "verse", "outro"]
    else:
        cuts   = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels = ["intro", "verse", "chorus", "verse", "chorus", "outro"]

    segs = []
    for i, lbl in enumerate(labels):
        s = round(full_duration * cuts[i],   1)
        e = round(full_duration * cuts[i+1], 1)
        segs.append({"label": lbl, "start": s, "end": e, "duration": round(e - s, 1)})
    return segs


def extract(raw_bytes: bytes, filename: str = "audio") -> dict[str, Any]:
    """
    Two-pass feature extraction:
      Pass 1 — instant: get real file duration via soundfile metadata.
      Pass 2 — analysis: load up to ANALYSIS_DUR seconds for all spectral work.
    Structure and duration_sec always reflect the full file.
    """
    import librosa

    log.info("feature_extraction_start", filename=filename)

    # ── Pass 1: real duration ──────────────────────────────────────────────
    full_duration = _get_full_duration(raw_bytes)
    log.info("duration_detected", full_duration=round(full_duration, 2) if full_duration else "unknown")

    # ── Pass 2: load analysis window ──────────────────────────────────────
    buf = io.BytesIO(raw_bytes)
    y, sr = librosa.load(buf, sr=SR, mono=True, duration=ANALYSIS_DUR)
    analysis_duration = len(y) / sr

    # If soundfile failed (e.g. MP3 edge case), fall back to librosa's duration
    if full_duration is None:
        log.warning("duration_fallback", note="soundfile failed, loading full file")
        buf2 = io.BytesIO(raw_bytes)
        y_full, _ = librosa.load(buf2, sr=SR, mono=True)
        full_duration = len(y_full) / sr

    f: dict[str, Any] = {}

    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LEN)
    f["bpm"]        = round(float(np.squeeze(tempo)), 2)
    f["beat_count"] = int(len(beats))

    # Key / Camelot
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    chroma_mean = chroma.mean(axis=1)
    key_idx, is_major, key_conf = _detect_key(chroma_mean)
    f["key"]            = KEYS[key_idx]
    f["scale"]          = "major" if is_major else "minor"
    f["key_confidence"] = key_conf
    f["camelot"]        = CAMELOT.get((key_idx, is_major), "?")

    # MFCC + Chroma stats
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LEN)
    f["mfcc_stats"]   = {"mean": mfcc.mean(axis=1).tolist(), "std": mfcc.std(axis=1).tolist()}
    f["chroma_stats"] = {"mean": chroma_mean.tolist(),       "std": chroma.std(axis=1).tolist()}

    # Spectral
    f["spectral_centroid_mean"]  = round(float(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LEN).mean()), 2)
    f["spectral_rolloff_mean"]   = round(float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()), 2)
    f["spectral_bandwidth_mean"] = round(float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()), 2)
    f["spectral_contrast_mean"]  = round(float(librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LEN).mean()), 2)
    f["zero_crossing_rate_mean"] = round(float(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN).mean()), 5)

    # Energy / Dynamics
    rms = librosa.feature.rms(y=y, hop_length=HOP_LEN)
    f["energy"]        = round(float(rms.mean()), 5)
    f["loudness_lufs"] = round(float(20 * np.log10(rms.mean() + 1e-9)), 2)
    f["dynamic_range"] = round(float(20 * np.log10((rms.max() + 1e-9) / (rms.min() + 1e-9))), 2)

    # Danceability + Groove
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=HOP_LEN)
    f["danceability"]  = round(float(np.clip(pulse.mean() * 3, 0, 1)), 3)
    f["beat_strength"] = round(float(np.clip(onset_env.mean() / 5.0, 0, 1)), 3)

    if len(beats) >= 4:
        bt   = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LEN)
        diff = np.diff(bt)
        even = diff[::2].mean()
        odd  = diff[1::2].mean() if len(diff[1::2]) else even
        f["swing_ratio"]     = round(float(even / (odd + 1e-8)), 3)
        ideal                = 60.0 / float(np.squeeze(tempo))
        f["beat_regularity"] = round(float(np.clip(1.0 - np.std(diff) / (ideal + 1e-8), 0, 1)), 3)
    else:
        f["swing_ratio"]     = 1.0
        f["beat_regularity"] = 0.5

    hist = np.histogram(onset_env, bins=20)[0].astype(float)
    hist /= hist.sum() + 1e-8
    f["rhythmic_complexity"] = round(float(np.clip(-np.sum(hist * np.log2(hist + 1e-8)) / np.log2(20), 0, 1)), 3)
    f["groove_feel"]         = "swung" if f["swing_ratio"] > 1.15 else "straight" if f["swing_ratio"] < 0.9 else "even"

    # Chords
    prog, funcs, chords = _chords(chroma_mean, key_idx)
    f["chord_progression"]  = prog
    f["harmonic_functions"] = funcs
    f["chords"]             = chords

    # ── Structure — always based on full file duration ─────────────────────
    f["segments"]     = _structure(full_duration, float(np.squeeze(tempo)))
    f["section_count"] = len(f["segments"])
    f["duration_sec"] = round(full_duration, 2)

    # ML feature vector (based on analysis window — intentional)
    f["feature_vector"] = [round(float(v), 6) for v in (
        f["mfcc_stats"]["mean"] + f["mfcc_stats"]["std"] + f["chroma_stats"]["mean"] + [
            f["bpm"], f["spectral_centroid_mean"], f["spectral_rolloff_mean"],
            f["spectral_bandwidth_mean"], f["spectral_contrast_mean"],
            f["zero_crossing_rate_mean"], f["energy"],
            f["danceability"], f["beat_strength"], f["rhythmic_complexity"],
        ]
    )]

    log.info("feature_extraction_complete",
             bpm=f["bpm"], key=f["key"], camelot=f["camelot"],
             duration_full=f["duration_sec"], duration_analysed=round(analysis_duration, 2))
    return f
