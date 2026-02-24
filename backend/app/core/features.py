import io
from typing import Any
import numpy as np
import structlog

log = structlog.get_logger()

N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
MAX_DURATION_SEC = 60

# Camelot wheel mapping: (key_index, is_major) -> camelot
CAMELOT = {
    (0,  True):  "8B",  (1,  True):  "3B",  (2,  True):  "10B",
    (3,  True):  "5B",  (4,  True):  "12B", (5,  True):  "7B",
    (6,  True):  "2B",  (7,  True):  "9B",  (8,  True):  "4B",
    (9,  True):  "11B", (10, True):  "6B",  (11, True):  "1B",
    (0,  False): "5A",  (1,  False): "12A", (2,  False): "7A",
    (3,  False): "2A",  (4,  False): "9A",  (5,  False): "4A",
    (6,  False): "11A", (7,  False): "6A",  (8,  False): "1A",
    (9,  False): "8A",  (10, False): "3A",  (11, False): "10A",
}

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord templates (major, minor, dom7, maj7, min7)
CHORD_TEMPLATES = {}
for i, k in enumerate(KEYS):
    major = np.zeros(12); major[[i, (i+4)%12, (i+7)%12]] = 1
    minor = np.zeros(12); minor[[i, (i+3)%12, (i+7)%12]] = 1
    dom7  = np.zeros(12); dom7[[i, (i+4)%12, (i+7)%12, (i+10)%12]] = 1
    maj7  = np.zeros(12); maj7[[i, (i+4)%12, (i+7)%12, (i+11)%12]] = 1
    min7  = np.zeros(12); min7[[i, (i+3)%12, (i+7)%12, (i+10)%12]] = 1
    CHORD_TEMPLATES[k + "maj"]  = major
    CHORD_TEMPLATES[k + "min"]  = minor
    CHORD_TEMPLATES[k + "7"]    = dom7
    CHORD_TEMPLATES[k + "maj7"] = maj7
    CHORD_TEMPLATES[k + "min7"] = min7

NUMERAL_MAP = {
    0: ("I",   "Imaj7"),
    2: ("II",  "IImin7"),
    3: ("bIII","bIIImaj7"),
    4: ("III", "IIImin7"),
    5: ("IV",  "IVmaj7"),
    6: ("bV",  "bVmaj7"),
    7: ("V",   "V7"),
    8: ("bVI", "bVImaj7"),
    9: ("VI",  "VImin7"),
    10:("bVII","bVII7"),
    11:("VII", "VIImin7"),
}


def _detect_key(chroma_mean: np.ndarray) -> tuple[int, bool, float]:
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    major_corr = [np.corrcoef(np.roll(major_profile,-i), chroma_mean)[0,1] for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(minor_profile,-i), chroma_mean)[0,1] for i in range(12)]
    best_major_idx = int(np.argmax(major_corr))
    best_minor_idx = int(np.argmax(minor_corr))
    if major_corr[best_major_idx] >= minor_corr[best_minor_idx]:
        return best_major_idx, True, round(float(major_corr[best_major_idx]), 3)
    else:
        return best_minor_idx, False, round(float(minor_corr[best_minor_idx]), 3)


def _detect_chords(y: np.ndarray, sr: int, key_idx: int) -> dict:
    """Detect chord progression and harmonic functions."""
    try:
        import librosa
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH*4)
        n_frames = chroma.shape[1]
        segment_size = max(1, n_frames // 8)

        chord_sequence = []
        for i in range(0, n_frames, segment_size):
            segment = chroma[:, i:i+segment_size].mean(axis=1)
            best_chord, best_score = None, -1
            for chord_name, template in CHORD_TEMPLATES.items():
                score = float(np.dot(template / (np.linalg.norm(template) + 1e-8),
                                     segment / (np.linalg.norm(segment) + 1e-8)))
                if score > best_score:
                    best_score = score
                    best_chord = chord_name

            if best_chord:
                root = best_chord.replace("maj7","").replace("min7","").replace("maj","").replace("min","").replace("7","")
                root_idx = KEYS.index(root) if root in KEYS else 0
                interval = (root_idx - key_idx) % 12
                numeral, numeral7 = NUMERAL_MAP.get(interval, ("?", "?"))
                quality = "maj7" if "maj7" in best_chord else "min7" if "min7" in best_chord else "7" if "7" in best_chord else "maj" if "maj" in best_chord else "min"
                func = numeral7 if "7" in quality else numeral
                chord_sequence.append({"chord": best_chord, "function": func})

        # Deduplicate consecutive same chords
        deduped = []
        for c in chord_sequence:
            if not deduped or deduped[-1]["chord"] != c["chord"]:
                deduped.append(c)

        progression = " — ".join([c["chord"] for c in deduped[:8]])
        functions   = " — ".join([c["function"] for c in deduped[:8]])
        return {"progression": progression, "functions": functions, "chords": deduped[:8]}
    except Exception as e:
        log.warning("chord_detection_failed", error=str(e))
        return {"progression": "", "functions": "", "chords": []}


def _detect_structure(y: np.ndarray, sr: int) -> dict:
    """Detect intro/verse/chorus segments using recurrence matrix."""
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH*4)
        R = librosa.segment.recurrence_matrix(mfcc, mode="affinity", sym=True)
        bounds = librosa.segment.agglomerative(R, 6)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP_LENGTH*4)

        labels = ["intro", "verse", "chorus", "verse", "chorus", "outro"]
        segments = []
        for i, (start, end) in enumerate(zip(bound_times[:-1], bound_times[1:])):
            segments.append({
                "label": labels[min(i, len(labels)-1)],
                "start": round(float(start), 1),
                "end":   round(float(end), 1),
                "duration": round(float(end - start), 1),
            })
        return {"segments": segments, "section_count": len(segments)}
    except Exception as e:
        log.warning("structure_detection_failed", error=str(e))
        return {"segments": [], "section_count": 0}


def _detect_groove(y: np.ndarray, sr: int, tempo: float, beats: np.ndarray) -> dict:
    """Measure swing ratio and beat strength."""
    try:
        import librosa
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        beat_strength = float(np.mean(onset_env[beats])) if len(beats) > 0 else 0.0

        # Swing: ratio of even/odd 8th note gaps
        if len(beats) >= 4:
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
            diffs = np.diff(beat_times)
            if len(diffs) >= 2:
                even = diffs[::2].mean()
                odd  = diffs[1::2].mean() if len(diffs[1::2]) > 0 else even
                swing_ratio = round(float(even / (odd + 1e-8)), 3)
            else:
                swing_ratio = 1.0
        else:
            swing_ratio = 1.0

        # Rhythmic complexity: onset entropy
        onset_hist = np.histogram(onset_env, bins=20)[0].astype(float)
        onset_hist = onset_hist / (onset_hist.sum() + 1e-8)
        complexity = float(-np.sum(onset_hist * np.log2(onset_hist + 1e-8))) / np.log2(20)

        # Beat regularity
        if len(beats) >= 4:
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
            ideal_period = 60.0 / tempo
            actual_diffs = np.diff(beat_times)
            regularity = 1.0 - float(np.std(actual_diffs) / (ideal_period + 1e-8))
            regularity = float(np.clip(regularity, 0, 1))
        else:
            regularity = 0.5

        return {
            "beat_strength":       round(float(np.clip(beat_strength / 5.0, 0, 1)), 3),
            "swing_ratio":         swing_ratio,
            "rhythmic_complexity": round(float(np.clip(complexity, 0, 1)), 3),
            "beat_regularity":     round(regularity, 3),
            "groove_feel":         "swung" if swing_ratio > 1.15 else "straight" if swing_ratio < 0.9 else "even",
        }
    except Exception as e:
        log.warning("groove_detection_failed", error=str(e))
        return {"beat_strength": 0, "swing_ratio": 1.0, "rhythmic_complexity": 0, "beat_regularity": 0, "groove_feel": "unknown"}


class FeatureExtractor:

    async def extract(self, wav_bytes: bytes) -> dict[str, Any]:
        import librosa

        buf = io.BytesIO(wav_bytes)
        y, sr = librosa.load(buf, sr=22050, mono=True, duration=MAX_DURATION_SEC)
        log.info("feature_extraction_start", duration=round(len(y)/sr, 2))

        features: dict[str, Any] = {}

        # ── BPM & Beats ──────────────────────────────────────────────────
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["bpm"]        = round(float(tempo), 2)
        features["beat_count"] = int(len(beats))

        # ── Key, Scale, Camelot ──────────────────────────────────────────
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma_mean = chroma.mean(axis=1)
        key_idx, is_major, key_conf = _detect_key(chroma_mean)
        features["key"]            = KEYS[key_idx]
        features["scale"]          = "major" if is_major else "minor"
        features["key_confidence"] = key_conf
        features["camelot"]        = CAMELOT.get((key_idx, is_major), "?")

        # ── MFCC ─────────────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        features["mfcc_stats"] = {
            "mean": mfcc.mean(axis=1).tolist(),
            "std":  mfcc.std(axis=1).tolist(),
        }
        features["chroma_stats"] = {
            "mean": chroma_mean.tolist(),
            "std":  chroma.std(axis=1).tolist(),
        }

        # ── Spectral ─────────────────────────────────────────────────────
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["spectral_centroid_mean"] = round(float(spec_centroid.mean()), 2)
        features["spectral_centroid_std"]  = round(float(spec_centroid.std()), 2)

        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features["spectral_rolloff_mean"]  = round(float(spec_rolloff.mean()), 2)

        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = round(float(spec_bandwidth.mean()), 2)

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["spectral_contrast_mean"] = round(float(spec_contrast.mean()), 2)

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        features["zero_crossing_rate_mean"] = round(float(zcr.mean()), 5)

        # ── Energy & Dynamics ────────────────────────────────────────────
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        features["energy"]        = round(float(rms.mean()), 5)
        features["energy_std"]    = round(float(rms.std()), 5)
        features["loudness_lufs"] = round(float(20 * np.log10(rms.mean() + 1e-9)), 2)
        features["dynamic_range"] = round(float(20 * np.log10((rms.max() + 1e-9) / (rms.min() + 1e-9))), 2)
        features["crest_factor"]  = round(float(np.max(np.abs(y)) / (rms.mean() + 1e-9)), 2)

        # ── Danceability ─────────────────────────────────────────────────
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        features["danceability"] = round(float(np.clip(pulse.mean() * 3, 0, 1)), 3)

        # ── Groove & Rhythm ──────────────────────────────────────────────
        groove = _detect_groove(y, sr, float(tempo), beats)
        features.update(groove)

        # ── Chord Progression (max 15s) ──────────────────────────────────
        import asyncio, concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                chords = await asyncio.wait_for(
                    loop.run_in_executor(pool, _detect_chords, y, sr, key_idx),
                    timeout=15.0
                )
        except asyncio.TimeoutError:
            log.warning("chord_detection_timeout")
            chords = {"progression": "", "functions": "", "chords": []}
        except Exception as e:
            log.warning("chord_detection_error", error=str(e))
            chords = {"progression": "", "functions": "", "chords": []}
        features["chord_progression"]  = chords.get("progression", "")
        features["harmonic_functions"]  = chords.get("functions", "")
        features["chords"]              = chords.get("chords", [])

        # ── Structure (max 10s) ───────────────────────────────────────────
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                structure = await asyncio.wait_for(
                    loop.run_in_executor(pool, _detect_structure, y, sr),
                    timeout=10.0
                )
        except asyncio.TimeoutError:
            log.warning("structure_detection_timeout")
            structure = {"segments": [], "section_count": 0}
        except Exception as e:
            log.warning("structure_detection_error", error=str(e))
            structure = {"segments": [], "section_count": 0}
        features["segments"]      = structure.get("segments", [])
        features["section_count"] = structure.get("section_count", 0)

        # ── Feature vector for ML ────────────────────────────────────────
        features["feature_vector"] = [round(float(v), 6) for v in (
            features["mfcc_stats"]["mean"]
            + features["mfcc_stats"]["std"]
            + features["chroma_stats"]["mean"]
            + [
                features["bpm"],
                features["spectral_centroid_mean"],
                features["spectral_rolloff_mean"],
                features["spectral_bandwidth_mean"],
                features["spectral_contrast_mean"],
                features["zero_crossing_rate_mean"],
                features["energy"],
                features["danceability"],
                features["beat_strength"],
                features["rhythmic_complexity"],
            ]
        )]

        log.info("feature_extraction_complete", bpm=features["bpm"], key=features["key"], camelot=features["camelot"])
        return features
