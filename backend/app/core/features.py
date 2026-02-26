"""
Pure synchronous feature extraction — no async, no event loop issues.
Called from a ThreadPoolExecutor in tracks.py.

Strategy (Alt 2 — two-pass):
  Pass 1: Read full duration instantly via soundfile (no decode).
  Pass 2: Load up to ANALYSIS_DUR seconds at low SR for spectral/rhythm work.
  Structure and duration_sec always reflect the full file.

Key detection uses a three-method ensemble:
  1. Chroma + cosine similarity (Krumhansl-Schmuckler profiles)
  2. Harmonic fit — how well each chord fits each of 24 keys
  3. Melodic f0 via pyin — pitch histogram weighted by beat position
  Each method contributes with calibrated structural weights.
  Falls back gracefully when a method has insufficient signal.
"""
import io
from typing import Any
import numpy as np
import structlog

log = structlog.get_logger()

# ── Constants ──────────────────────────────────────────────────────────────────

N_MFCC       = 13
HOP_LEN      = 1024
N_FFT        = 2048
ANALYSIS_DUR = 60    # seconds of audio used for spectral/rhythm analysis
SR           = 16000

# Camelot wheel: (key_index 0-11, is_major) -> Camelot notation
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

# Krumhansl-Schmuckler key profiles (module-level, computed once at import)
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# Pre-normalised for cosine similarity — avoids repeated division in inner loops
_KS_MAJOR_NORM = _KS_MAJOR / np.linalg.norm(_KS_MAJOR)
_KS_MINOR_NORM = _KS_MINOR / np.linalg.norm(_KS_MINOR)

# Harmonic fit scores
_SCORE_TONIC     = 3.0   # grade I / i
_SCORE_DOMINANT  = 2.5   # grade V
_SCORE_DIATONIC  = 2.0   # other diatonic chords
_SCORE_BORROW    = 1.0   # common borrowed / modal chords
_SCORE_CHROMATIC = 0.2   # outside the key

# Diatonic degree -> expected chord quality
_MAJOR_DEGREES = {0:"maj",2:"min",4:"min",5:"maj",7:"maj",9:"min",11:"dim"}
_MINOR_DEGREES = {2:"dim",3:"maj",5:"min",7:"min",8:"maj",10:"maj"}


# ── Utility ────────────────────────────────────────────────────────────────────

def _get_full_duration(raw_bytes: bytes) -> float | None:
    """
    Read the file's real duration without decoding audio.
    Falls back to None if soundfile cannot handle the format.
    """
    try:
        import soundfile as sf
        buf  = io.BytesIO(raw_bytes)
        info = sf.info(buf)
        return info.duration
    except Exception:
        return None


# ── Key detection — Method 1: Chroma + cosine ─────────────────────────────────

def _detect_key_chroma(chroma_mean: np.ndarray) -> tuple[int, bool, float]:
    """
    Cosine similarity between the chroma mean vector and Krumhansl-Schmuckler
    key profiles for all 24 keys.

    Roll direction: np.roll(profile, +i) places the root weight at index i,
    so key root i gets the highest KS weight (6.33/6.35).  The bug in previous
    versions used -i which inverted this mapping.

    Tonic-first tiebreaker: when the strongest chroma note agrees with the
    detected root, the major/minor choice is refined by comparing the minor
    third (root+3) against the major third (root+4).

    Returns (key_idx 0-11, is_major, cosine_score 0..1).
    """
    cm   = np.array(chroma_mean, dtype=float)
    norm = np.linalg.norm(cm)
    if norm < 1e-8:
        return 0, True, 0.0
    cm_norm = cm / norm

    best_key, best_major, best_score = 0, True, -np.inf
    for i in range(12):
        maj_s = float(np.dot(np.roll(_KS_MAJOR_NORM, i), cm_norm))
        min_s = float(np.dot(np.roll(_KS_MINOR_NORM, i), cm_norm))
        if maj_s > best_score:
            best_score, best_key, best_major = maj_s, i, True
        if min_s > best_score:
            best_score, best_key, best_major = min_s, i, False

    # Tonic-first tiebreaker
    tonic_idx = int(np.argmax(cm))
    if tonic_idx == best_key:
        minor_third = float(cm[(tonic_idx + 3) % 12])
        major_third = float(cm[(tonic_idx + 4) % 12])
        if abs(minor_third - major_third) > 0.01:
            best_major = major_third > minor_third

    return best_key, best_major, round(best_score, 3)


# ── Key detection — Method 2: Harmonic fit ────────────────────────────────────

def _chord_fit_score(chord_root: int, chord_quality: str,
                     key_root: int, is_major: bool) -> float:
    """How well does one chord fit into the given key?"""
    interval = (chord_root - key_root) % 12

    if is_major:
        if interval == 0:  return _SCORE_TONIC
        if interval == 7:  return _SCORE_DOMINANT
        expected = _MAJOR_DEGREES.get(interval)
        if expected == chord_quality:              return _SCORE_DIATONIC
        if expected == "dim" and chord_quality == "min": return _SCORE_BORROW
        # Common borrows: bVII(10) maj, bIII(3) maj, iv(5) min
        if interval in (10, 3) and chord_quality == "maj": return _SCORE_BORROW
        if interval == 5 and chord_quality == "min":       return _SCORE_BORROW
        return _SCORE_CHROMATIC

    else:  # minor key
        if interval == 0:  return _SCORE_TONIC
        # Harmonic minor V: major chord a fifth above the root
        if interval == 7 and chord_quality == "maj": return _SCORE_DOMINANT
        # Natural minor v: minor chord a fifth above the root
        if interval == 7 and chord_quality == "min": return _SCORE_DIATONIC
        expected = _MINOR_DEGREES.get(interval)
        if expected == chord_quality:              return _SCORE_DIATONIC
        # Common borrows: IV maj(5), II min(2)
        if interval in (5, 2) and chord_quality == "min": return _SCORE_BORROW
        return _SCORE_CHROMATIC


def _extract_chord_sequence(chroma: np.ndarray,
                             n_frames_per_chord: int = 30) -> list[str]:
    """
    Segment the chroma matrix into overlapping windows and identify the
    best-matching major or minor chord template for each window.

    Args:
        chroma: (12, T) chroma matrix
        n_frames_per_chord: window size in frames (~2 s at SR=16000, HOP=1024)

    Returns:
        List of chord labels e.g. ['G_min', 'F_maj', 'D#_maj', ...].
        Consecutive duplicates are collapsed.
    """
    n_frames = chroma.shape[1]
    if n_frames < n_frames_per_chord:
        windows = [chroma]
    else:
        step    = max(1, n_frames_per_chord // 2)   # 50 % overlap
        windows = [
            chroma[:, i : i + n_frames_per_chord]
            for i in range(0, n_frames - n_frames_per_chord + 1, step)
        ]

    sequence: list[str] = []
    for win in windows:
        cm      = win.mean(axis=1)
        cm_norm = cm / (np.linalg.norm(cm) + 1e-8)

        best_root, best_quality, best_score = 0, "maj", -np.inf
        for root in range(12):
            # Major triad template: root 1.0, major-third 0.8, fifth 0.9
            t_maj = np.zeros(12)
            t_maj[root]           = 1.0
            t_maj[(root+4) % 12]  = 0.8
            t_maj[(root+7) % 12]  = 0.9
            t_maj /= np.linalg.norm(t_maj)
            s_maj = float(np.dot(t_maj, cm_norm))

            # Minor triad template: root 1.0, minor-third 0.8, fifth 0.9
            t_min = np.zeros(12)
            t_min[root]           = 1.0
            t_min[(root+3) % 12]  = 0.8
            t_min[(root+7) % 12]  = 0.9
            t_min /= np.linalg.norm(t_min)
            s_min = float(np.dot(t_min, cm_norm))

            if s_maj > best_score:
                best_score, best_root, best_quality = s_maj, root, "maj"
            if s_min > best_score:
                best_score, best_root, best_quality = s_min, root, "min"

        sequence.append(f"{KEYS[best_root]}_{best_quality}")

    # Collapse consecutive duplicates (A-A-B-B -> A-B)
    collapsed: list[str] = []
    for chord in sequence:
        if not collapsed or chord != collapsed[-1]:
            collapsed.append(chord)
    return collapsed


def _detect_key_harmonic(chord_sequence: list[str]) -> tuple[int, bool, float] | None:
    """
    Find the key that maximises the mean diatonic fit score across all chords.
    The first and last chords receive a tonic-position bonus when they match
    the candidate root — useful for short loop-based progressions.

    Returns (key_idx, is_major, confidence) or None if sequence too short.
    Confidence is the normalised margin between the best and second-best key.
    """
    if len(chord_sequence) < 2:
        return None

    parsed: list[tuple[int, str]] = []
    for label in chord_sequence:
        parts = label.split("_")
        if len(parts) == 2 and parts[0] in KEYS:
            parsed.append((KEYS.index(parts[0]), parts[1]))

    if len(parsed) < 2:
        return None

    all_scores: list[tuple[float, int, bool]] = []
    for key_root in range(12):
        for is_major in (True, False):
            total = 0.0
            for pos, (cr, cq) in enumerate(parsed):
                fit = _chord_fit_score(cr, cq, key_root, is_major)
                # Bonus when first or last chord is the tonic
                if pos in (0, len(parsed) - 1) and (cr - key_root) % 12 == 0:
                    fit += 0.8
                total += fit
            all_scores.append((total / len(parsed), key_root, is_major))

    all_scores.sort(reverse=True)
    best_score, best_key, best_major = all_scores[0]
    second_score = all_scores[1][0]

    margin = best_score - second_score
    conf   = round(float(min(1.0, margin / _SCORE_TONIC + 0.3)), 3)

    return best_key, best_major, conf


# ── Key detection — Method 3: Melodic f0 via pyin ─────────────────────────────

def _detect_key_melody(
    y: np.ndarray,
    sr: int,
    beat_frames: np.ndarray | None = None,
    hop_length: int = HOP_LEN,
) -> tuple[int, bool, float, float]:
    """
    Detect key from the melodic pitch content via pyin f0-tracking.

    Pitch-class histogram is weighted by:
      - pyin voiced probability (per-frame confidence)
      - beat-position bonus: frames landing on a beat get 1.5x weight

    Returns (key_idx, is_major, key_confidence, melody_weight).
    melody_weight ~ mean voiced probability; near 0 means no detectable
    melody (percussion, noise) and the caller should ignore the result.
    """
    import librosa

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )
    except Exception:
        return 0, True, 0.0, 0.0

    melody_weight = float(np.mean(voiced_probs))

    if melody_weight < 0.15:
        return 0, True, 0.0, round(melody_weight, 3)

    beat_set = set(beat_frames.tolist()) if beat_frames is not None else set()

    pitch_histogram = np.zeros(12)
    for idx in np.where(voiced_flag & ~np.isnan(f0))[0]:
        prob = float(voiced_probs[idx])
        pc   = int(round(librosa.hz_to_midi(f0[idx]))) % 12
        pitch_histogram[pc] += prob * (1.5 if idx in beat_set else 1.0)

    total = pitch_histogram.sum()
    if total < 1e-6:
        return 0, True, 0.0, round(melody_weight, 3)

    ph_norm = pitch_histogram / total
    ph_norm = ph_norm / (np.linalg.norm(ph_norm) + 1e-8)

    best_key, best_major, best_score = 0, True, -np.inf
    for i in range(12):
        maj_s = float(np.dot(np.roll(_KS_MAJOR_NORM, i), ph_norm))
        min_s = float(np.dot(np.roll(_KS_MINOR_NORM, i), ph_norm))
        if maj_s > best_score:
            best_score, best_key, best_major = maj_s, i, True
        if min_s > best_score:
            best_score, best_key, best_major = min_s, i, False

    return best_key, best_major, round(best_score, 3), round(melody_weight, 3)


# ── Key detection — Ensemble combiner ─────────────────────────────────────────

def _combine_key_estimates(
    chroma_result:   tuple[int, bool, float],
    harmonic_result: tuple[int, bool, float] | None,
    melody_result:   tuple[int, bool, float, float] | None,
) -> tuple[int, bool, float]:
    """
    Weighted ensemble of the three key estimates.

    Structural weights (rationale in comments):
      Chroma   1.0 — always active; good tonal balance, poor on relative keys
      Harmonic 3.0 — active when chords available; knows which chords appear
                     Weight raised to 3.0: when chroma has a single very dominant
                     note (e.g. A=0.923) cosine similarity reaches ~0.95, which at
                     weight 2.0 beat a correct harmonic signal of conf~0.45.
      Melodic  3.0 * melody_weight — strongest signal but requires clear melody

    Returns (key_idx, is_major, margin_confidence 0..1).
    Confidence is the fractional gap between the top and second-best slot.
    """
    scores = np.zeros(24)  # 0-11 = major, 12-23 = minor

    def cast_vote(key_idx: int, is_major: bool, conf: float, weight: float) -> None:
        slot = key_idx if is_major else key_idx + 12
        scores[slot] += conf * weight

    # Method 1 — always present
    cast_vote(*chroma_result[:3], weight=1.0)

    # Method 2 — when harmonic analysis succeeded
    if harmonic_result is not None:
        cast_vote(*harmonic_result[:3], weight=3.0)

    # Method 3 — when sufficient melodic content exists
    if melody_result is not None:
        ki, im, mc, mw = melody_result
        if mw > 0.15:
            cast_vote(ki, im, mc, weight=3.0 * mw)

    best = int(np.argmax(scores))
    is_major_result = best < 12
    key_result      = best % 12

    sorted_s = np.sort(scores)[::-1]
    margin   = float((sorted_s[0] - sorted_s[1]) / (sorted_s[0] + 1e-8))

    # Tie-breaker: when ensemble margin < 0.15 AND chroma cosine < 0.90
    # (indicating the chroma signal is ambiguous), and harmonic disagrees with
    # the current winner, override with harmonic.
    # Guard: chroma_result[2] >= 0.90 → chroma is authoritative, do NOT override.
    # Rationale: chroma cosine suffers from semitone bleed (F vs F# adjacent bins),
    # causing E_min to beat A_min even when F natural is present (F∈A_min, F∉E_min).
    # Harmonic chord analysis is immune to single-semitone bleed.
    chroma_cosine = chroma_result[2]
    if margin < 0.15 and chroma_cosine < 0.90 and harmonic_result is not None:
        hk, hm = harmonic_result[0], harmonic_result[1]
        if (hk != key_result) or (hm != is_major_result):
            key_result      = hk
            is_major_result = hm
            margin          = 0.0  # signals forced tie-break

    return key_result, is_major_result, round(margin, 3)


# ── Legacy output helpers (unchanged from v1) ──────────────────────────────────

def _chords(chroma_mean: np.ndarray, key_idx: int) -> tuple:
    """
    Build chord-progression label and chord list for the API response.
    Uses the global chroma mean — not time-segmented.
    """
    top       = np.argsort(chroma_mean)[::-1][:4]
    intervals = sorted(int(n - key_idx) % 12 for n in top)
    best, best_score = "I-IV-V", 0
    for name, pattern in COMMON_PROGRESSIONS.items():
        score = sum(1 for p in pattern if p in intervals)
        if score > best_score:
            best_score, best = score, name
    chords = [{"chord": KEYS[(key_idx+i) % 12], "function": NUMERAL.get(i, "?")} for i in intervals]
    return best, " — ".join(NUMERAL.get(i, "?") for i in intervals), chords


def _structure(full_duration: float, bpm: float) -> list:
    """
    Build song structure segments based on the FULL file duration.
    Timestamps are always correct regardless of analysis window length.
    """
    bar  = (60.0 / max(float(bpm), 60)) * 4
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


# ── Main entry point ───────────────────────────────────────────────────────────

def extract(raw_bytes: bytes, filename: str = "audio") -> dict[str, Any]:
    """
    Two-pass feature extraction:
      Pass 1 — instant: real file duration via soundfile metadata.
      Pass 2 — analysis: load up to ANALYSIS_DUR seconds for spectral work.
    Structure and duration_sec always reflect the full file.
    """
    import librosa

    log.info("feature_extraction_start", filename=filename)

    # ── Pass 1: real duration ────────────────────────────────────────────────
    full_duration = _get_full_duration(raw_bytes)
    log.info("duration_detected",
             full_duration=round(full_duration, 2) if full_duration else "unknown")

    # ── Pass 2: load analysis window ─────────────────────────────────────────
    buf    = io.BytesIO(raw_bytes)
    y, sr  = librosa.load(buf, sr=SR, mono=True, duration=ANALYSIS_DUR)
    analysis_duration = len(y) / sr

    if full_duration is None:
        log.warning("duration_fallback", note="soundfile failed, loading full file")
        buf2 = io.BytesIO(raw_bytes)
        y_full, _ = librosa.load(buf2, sr=SR, mono=True)
        full_duration = len(y_full) / sr

    f: dict[str, Any] = {}

    # ── BPM — multi-candidate median ─────────────────────────────────────────
    # onset_env is shared with danceability / groove calculations below
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP_LEN, onset_envelope=onset_env
    )
    try:
        tempo_arr  = librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LEN, aggregate=None
        )
        bpm_candidates = [round(float(t), 1) for t in tempo_arr[:4]]
        bpm_val        = float(np.median(tempo_arr))
    except Exception:
        bpm_candidates = [round(float(np.squeeze(tempo)), 1)]
        bpm_val        = float(np.squeeze(tempo))

    f["bpm"]        = round(bpm_val, 1)
    f["beat_count"] = int(len(beats))
    log.info("bpm_detected", bpm=f["bpm"], candidates=bpm_candidates)

    # ── Tuning ───────────────────────────────────────────────────────────────
    tuning_cents = round(float(librosa.estimate_tuning(y=y, sr=sr)) * 100, 1)
    log.info("tuning_estimated", tuning_cents=tuning_cents)

    # ── Chroma — computed ONCE, reused by all key methods ────────────────────
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    chroma_mean = chroma.mean(axis=1)

    top3_idx = np.argsort(chroma_mean)[::-1][:3]
    log.info("chroma_top3",
             notes={KEYS[i]: round(float(chroma_mean[i]), 3) for i in top3_idx})

    # ── Key detection — three-method ensemble ───────────────────────────────

    # Method 1: chroma cosine similarity (always runs)
    chroma_result = _detect_key_chroma(chroma_mean)

    # Method 2: harmonic fit via windowed chord sequence
    # Window width ~2 s: 2 * SR / HOP_LEN ≈ 31 frames at SR=16000, HOP=1024
    n_frames_per_chord = max(4, int(2.0 * sr / HOP_LEN))
    chord_sequence     = _extract_chord_sequence(chroma, n_frames_per_chord)
    harmonic_result    = (
        _detect_key_harmonic(chord_sequence)
        if len(chord_sequence) >= 2 else None
    )

    # Method 3: melodic f0 via pyin (gracefully inactive for rhythmic/noise content)
    melody_result = _detect_key_melody(y, sr, beat_frames=beats, hop_length=HOP_LEN)

    # Combine all three into a single key estimate
    key_idx, is_major, key_conf = _combine_key_estimates(
        chroma_result, harmonic_result, melody_result
    )

    f["key"]            = KEYS[key_idx]
    f["scale"]          = "major" if is_major else "minor"
    f["key_confidence"] = key_conf
    f["camelot"]        = CAMELOT.get((key_idx, is_major), "?")

    log.info("key_result",
             key=f["key"], scale=f["scale"][:3], conf=key_conf,
             chroma  = f"{KEYS[chroma_result[0]]}_{'maj' if chroma_result[1] else 'min'}",
             harmonic= (f"{KEYS[harmonic_result[0]]}_{'maj' if harmonic_result[1] else 'min'}"
                        if harmonic_result else "n/a"),
             melody  = (f"{KEYS[melody_result[0]]}_{'maj' if melody_result[1] else 'min'}"
                        if melody_result and melody_result[3] > 0.15 else "n/a"))

    # ── MFCC + Chroma stats ──────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LEN)
    f["mfcc_stats"]   = {"mean": mfcc.mean(axis=1).tolist(), "std": mfcc.std(axis=1).tolist()}
    f["chroma_stats"] = {"mean": chroma_mean.tolist(),       "std": chroma.std(axis=1).tolist()}

    # ── Spectral ─────────────────────────────────────────────────────────────
    f["spectral_centroid_mean"]  = round(float(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LEN).mean()), 2)
    f["spectral_rolloff_mean"]   = round(float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()), 2)
    f["spectral_bandwidth_mean"] = round(float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()), 2)
    f["spectral_contrast_mean"]  = round(float(librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LEN).mean()), 2)
    f["zero_crossing_rate_mean"] = round(float(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN).mean()), 5)

    # ── Energy / Dynamics ────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=HOP_LEN)
    f["energy"]        = round(float(rms.mean()), 5)
    f["loudness_lufs"] = round(float(20 * np.log10(rms.mean() + 1e-9)), 2)
    f["dynamic_range"] = round(float(20 * np.log10((rms.max() + 1e-9) / (rms.min() + 1e-9))), 2)

    # ── Danceability + Groove ─────────────────────────────────────────────────
    # onset_env already computed above — no second calculation needed
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=HOP_LEN)
    f["danceability"]  = round(float(np.clip(pulse.mean() * 3, 0, 1)), 3)
    f["beat_strength"] = round(float(np.clip(onset_env.mean() / 5.0, 0, 1)), 3)

    if len(beats) >= 4:
        bt   = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LEN)
        diff = np.diff(bt)
        even = diff[::2].mean()
        odd  = diff[1::2].mean() if len(diff[1::2]) else even
        f["swing_ratio"]     = round(float(even / (odd + 1e-8)), 3)
        ideal                = 60.0 / max(bpm_val, 1.0)
        f["beat_regularity"] = round(float(np.clip(1.0 - np.std(diff) / (ideal + 1e-8), 0, 1)), 3)
    else:
        f["swing_ratio"]     = 1.0
        f["beat_regularity"] = 0.5

    hist = np.histogram(onset_env, bins=20)[0].astype(float)
    hist /= hist.sum() + 1e-8
    f["rhythmic_complexity"] = round(float(np.clip(
        -np.sum(hist * np.log2(hist + 1e-8)) / np.log2(20), 0, 1)), 3)
    f["groove_feel"] = (
        "swung" if f["swing_ratio"] > 1.15
        else "straight" if f["swing_ratio"] < 0.9
        else "even"
    )

    # ── Chords for API output (uses global chroma mean) ──────────────────────
    prog, funcs, chords = _chords(chroma_mean, key_idx)
    f["chord_progression"]  = prog
    f["harmonic_functions"] = funcs
    f["chords"]             = chords

    # ── Structure — always based on full file duration ───────────────────────
    f["segments"]      = _structure(full_duration, bpm_val)
    f["section_count"] = len(f["segments"])
    f["duration_sec"]  = round(full_duration, 2)

    # ── ML feature vector (analysis window — intentional) ───────────────────
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
             duration_full=f["duration_sec"],
             duration_analysed=round(analysis_duration, 2))
    return f
