import io
from typing import Any
import numpy as np
import structlog

log = structlog.get_logger()

N_MFCC = 13
HOP_LENGTH = 1024   # Larger = faster
N_FFT = 2048
MAX_DURATION_SEC = 30  # Max 30s for speed

CAMELOT = {
    (0,True):"8B",(1,True):"3B",(2,True):"10B",(3,True):"5B",(4,True):"12B",(5,True):"7B",
    (6,True):"2B",(7,True):"9B",(8,True):"4B",(9,True):"11B",(10,True):"6B",(11,True):"1B",
    (0,False):"5A",(1,False):"12A",(2,False):"7A",(3,False):"2A",(4,False):"9A",(5,False):"4A",
    (6,False):"11A",(7,False):"6A",(8,False):"1A",(9,False):"8A",(10,False):"3A",(11,False):"10A",
}
KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Simple chord templates (triads only — much faster than full detection)
COMMON_PROGRESSIONS = {
    "I-IV-V":    [0, 5, 7],
    "I-V-vi-IV": [0, 7, 9, 5],
    "I-vi-IV-V": [0, 9, 5, 7],
    "ii-V-I":    [2, 7, 0],
    "I-IV-vi-V": [0, 5, 9, 7],
}


def _detect_key(chroma_mean):
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    major_corr = [np.corrcoef(np.roll(major_profile,-i), chroma_mean)[0,1] for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(minor_profile,-i), chroma_mean)[0,1] for i in range(12)]
    bmi = int(np.argmax(major_corr))
    bni = int(np.argmax(minor_corr))
    if major_corr[bmi] >= minor_corr[bni]:
        return bmi, True, round(float(major_corr[bmi]), 3)
    return bni, False, round(float(minor_corr[bni]), 3)


def _simple_chords(chroma_mean, key_idx):
    """Fast chord guess based on chroma profile — no segment analysis."""
    try:
        NUMERAL = {0:"I",2:"ii",3:"bIII",4:"III",5:"IV",6:"bV",7:"V",8:"bVI",9:"vi",10:"bVII",11:"vii"}
        # Top 4 most prominent notes
        top = np.argsort(chroma_mean)[::-1][:4]
        intervals = sorted([(int(n - key_idx) % 12) for n in top])

        # Match against common progressions
        best_prog, best_score = "I-IV-V", 0
        for name, pattern in COMMON_PROGRESSIONS.items():
            score = sum(1 for p in pattern if p in intervals)
            if score > best_score:
                best_score = score
                best_prog = name

        numerals = [NUMERAL.get(i, "?") for i in intervals[:4]]
        return {
            "progression": best_prog,
            "functions": " — ".join(numerals),
            "chords": [{"chord": KEYS[(key_idx + i) % 12], "function": NUMERAL.get(i,"?")} for i in intervals[:4]]
        }
    except Exception as e:
        log.warning("simple_chords_failed", error=str(e))
        return {"progression": "", "functions": "", "chords": []}


def _simple_structure(duration_sec, bpm):
    """Estimate structure from duration and BPM — no heavy computation."""
    try:
        beats_per_bar = 4
        bar_duration = (60.0 / max(bpm, 60)) * beats_per_bar
        bars = duration_sec / bar_duration

        # Typical pop structure proportions
        if bars < 16:
            segments = [
                {"label":"intro","start":0,"end":round(duration_sec*0.2,1),"duration":round(duration_sec*0.2,1)},
                {"label":"verse","start":round(duration_sec*0.2,1),"end":round(duration_sec*0.6,1),"duration":round(duration_sec*0.4,1)},
                {"label":"outro","start":round(duration_sec*0.6,1),"end":round(duration_sec,1),"duration":round(duration_sec*0.4,1)},
            ]
        else:
            segments = [
                {"label":"intro", "start":0,                       "end":round(duration_sec*0.1,1), "duration":round(duration_sec*0.1,1)},
                {"label":"verse", "start":round(duration_sec*0.1,1),"end":round(duration_sec*0.3,1),"duration":round(duration_sec*0.2,1)},
                {"label":"chorus","start":round(duration_sec*0.3,1),"end":round(duration_sec*0.5,1),"duration":round(duration_sec*0.2,1)},
                {"label":"verse", "start":round(duration_sec*0.5,1),"end":round(duration_sec*0.7,1),"duration":round(duration_sec*0.2,1)},
                {"label":"chorus","start":round(duration_sec*0.7,1),"end":round(duration_sec*0.9,1),"duration":round(duration_sec*0.2,1)},
                {"label":"outro", "start":round(duration_sec*0.9,1),"end":round(duration_sec,1),    "duration":round(duration_sec*0.1,1)},
            ]
        return {"segments": segments, "section_count": len(segments)}
    except Exception as e:
        log.warning("simple_structure_failed", error=str(e))
        return {"segments": [], "section_count": 0}


class FeatureExtractor:

    async def extract(self, wav_bytes: bytes) -> dict[str, Any]:
        import librosa

        buf = io.BytesIO(wav_bytes)
        # SR 16000 is much faster than 22050, sufficient for feature extraction
        y, sr = librosa.load(buf, sr=16000, mono=True, duration=MAX_DURATION_SEC)
        duration = len(y) / sr
        log.info("feature_extraction_start", duration=round(duration, 2))

        features: dict[str, Any] = {}

        # BPM & Beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["bpm"]        = round(float(tempo), 2)
        features["beat_count"] = int(len(beats))

        # Key / Scale / Camelot
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma_mean = chroma.mean(axis=1)
        key_idx, is_major, key_conf = _detect_key(chroma_mean)
        features["key"]            = KEYS[key_idx]
        features["scale"]          = "major" if is_major else "minor"
        features["key_confidence"] = key_conf
        features["camelot"]        = CAMELOT.get((key_idx, is_major), "?")

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        features["mfcc_stats"] = {
            "mean": mfcc.mean(axis=1).tolist(),
            "std":  mfcc.std(axis=1).tolist(),
        }
        features["chroma_stats"] = {
            "mean": chroma_mean.tolist(),
            "std":  chroma.std(axis=1).tolist(),
        }

        # Spectral
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["spectral_centroid_mean"]  = round(float(spec_centroid.mean()), 2)
        features["spectral_centroid_std"]   = round(float(spec_centroid.std()), 2)

        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features["spectral_rolloff_mean"]   = round(float(spec_rolloff.mean()), 2)

        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = round(float(spec_bandwidth.mean()), 2)

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["spectral_contrast_mean"]  = round(float(spec_contrast.mean()), 2)

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        features["zero_crossing_rate_mean"] = round(float(zcr.mean()), 5)

        # Energy & Dynamics
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        features["energy"]        = round(float(rms.mean()), 5)
        features["energy_std"]    = round(float(rms.std()), 5)
        features["loudness_lufs"] = round(float(20 * np.log10(rms.mean() + 1e-9)), 2)
        features["dynamic_range"] = round(float(20 * np.log10((rms.max() + 1e-9) / (rms.min() + 1e-9))), 2)
        features["crest_factor"]  = round(float(np.max(np.abs(y)) / (rms.mean() + 1e-9)), 2)

        # Danceability
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        features["danceability"] = round(float(np.clip(pulse.mean() * 3, 0, 1)), 3)

        # Groove (lightweight)
        features["beat_strength"] = round(float(np.clip(float(onset_env.mean()) / 5.0, 0, 1)), 3)

        if len(beats) >= 4:
            import librosa
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
            diffs = np.diff(beat_times)
            even = diffs[::2].mean() if len(diffs[::2]) > 0 else 1.0
            odd  = diffs[1::2].mean() if len(diffs[1::2]) > 0 else even
            swing_ratio = round(float(even / (odd + 1e-8)), 3)
            ideal = 60.0 / float(tempo)
            regularity = float(np.clip(1.0 - np.std(diffs) / (ideal + 1e-8), 0, 1))
        else:
            swing_ratio = 1.0
            regularity  = 0.5

        onset_hist = np.histogram(onset_env, bins=20)[0].astype(float)
        onset_hist /= (onset_hist.sum() + 1e-8)
        complexity = float(-np.sum(onset_hist * np.log2(onset_hist + 1e-8))) / np.log2(20)

        features["swing_ratio"]         = swing_ratio
        features["beat_regularity"]     = round(regularity, 3)
        features["rhythmic_complexity"] = round(float(np.clip(complexity, 0, 1)), 3)
        features["groove_feel"]         = "swung" if swing_ratio > 1.15 else "straight" if swing_ratio < 0.9 else "even"

        # Chords — fast heuristic, no segment analysis
        chords = _simple_chords(chroma_mean, key_idx)
        features["chord_progression"]  = chords.get("progression", "")
        features["harmonic_functions"] = chords.get("functions", "")
        features["chords"]             = chords.get("chords", [])

        # Structure — estimated from duration/BPM, no heavy computation
        structure = _simple_structure(duration, float(tempo))
        features["segments"]      = structure.get("segments", [])
        features["section_count"] = structure.get("section_count", 0)

        # Feature vector for ML
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
