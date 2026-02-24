import io
from typing import Any
import numpy as np
import structlog

log = structlog.get_logger()

N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
MAX_DURATION_SEC = 60  # Analysera max 60 sekunder


class FeatureExtractor:

    async def extract(self, wav_bytes: bytes) -> dict[str, Any]:
        import librosa

        buf = io.BytesIO(wav_bytes)
        # Ladda max 60 sekunder for snabbare analys
        y, sr = librosa.load(buf, sr=22050, mono=True, duration=MAX_DURATION_SEC)

        log.info("feature_extraction_start", duration=round(len(y) / sr, 2))

        features = {}

        # BPM
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["bpm"] = round(float(tempo), 2)
        features["beat_count"] = int(len(beats))

        # Tonart
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        key_idx = int(np.argmax(chroma.mean(axis=1)))
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        features["key"] = keys[key_idx]

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma_mean = chroma.mean(axis=1)
        rolled_major = [np.corrcoef(np.roll(major_profile, -i), chroma_mean)[0, 1] for i in range(12)]
        rolled_minor = [np.corrcoef(np.roll(minor_profile, -i), chroma_mean)[0, 1] for i in range(12)]
        features["scale"] = "major" if max(rolled_major) >= max(rolled_minor) else "minor"
        features["key_confidence"] = round(float(max(max(rolled_major), max(rolled_minor))), 3)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        features["mfcc_stats"] = {
            "mean": mfcc.mean(axis=1).tolist(),
            "std": mfcc.std(axis=1).tolist(),
            "min": mfcc.min(axis=1).tolist(),
            "max": mfcc.max(axis=1).tolist(),
        }

        features["chroma_stats"] = {
            "mean": chroma.mean(axis=1).tolist(),
            "std": chroma.std(axis=1).tolist(),
        }

        # Spektrala features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        features["spectral_centroid_mean"] = round(float(spectral_centroid.mean()), 2)
        features["spectral_centroid_std"] = round(float(spectral_centroid.std()), 2)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features["spectral_rolloff_mean"] = round(float(spectral_rolloff.mean()), 2)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = round(float(spectral_bandwidth.mean()), 2)

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        features["zero_crossing_rate_mean"] = round(float(zcr.mean()), 5)

        # Energi
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        features["energy"] = round(float(rms.mean()), 5)
        features["energy_std"] = round(float(rms.std()), 5)
        features["loudness_lufs"] = round(float(20 * np.log10(rms.mean() + 1e-9)), 2)

        # Danceability
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        features["danceability"] = round(float(np.clip(pulse.mean() * 3, 0, 1)), 3)

        # Feature-vektor
        feature_vector = (
            features["mfcc_stats"]["mean"]
            + features["mfcc_stats"]["std"]
            + features["chroma_stats"]["mean"]
            + [
                features["bpm"],
                features["spectral_centroid_mean"],
                features["spectral_rolloff_mean"],
                features["spectral_bandwidth_mean"],
                features["zero_crossing_rate_mean"],
                features["energy"],
                features["danceability"],
            ]
        )
        features["feature_vector"] = [round(float(v), 6) for v in feature_vector]

        log.info("feature_extraction_complete", bpm=features["bpm"], key=features["key"])
        return features
