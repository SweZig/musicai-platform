"""
Ingest-processor: validering, format-konvertering och normalisering.
Konverterar alla format till 44.1 kHz mono WAV 32-bit float.
"""
import io
import os
import tempfile
from typing import Tuple

import numpy as np
import soundfile as sf
import structlog

log = structlog.get_logger()

TARGET_SR = 44100
TARGET_CHANNELS = 1  # mono


class IngestProcessor:

    async def process(self, audio_bytes: bytes, filename: str) -> Tuple[bytes, dict]:
        """
        Bearbetar råa audio-bytes:
        1. Identifiera format
        2. Dekoda med pydub (hanterar MP3, AIFF, OGG etc.)
        3. Normalisera samplingfrekvens och kanaler
        4. Loudness-normalisering (EBU R128)
        5. Returnera WAV-bytes + metadata-dict
        """
        ext = os.path.splitext(filename)[1].lower()
        log.info("ingest_start", filename=filename, ext=ext, size=len(audio_bytes))

        try:
            audio_array, sr, info = self._decode(audio_bytes, ext)
        except Exception as e:
            raise ValueError(f"Kunde inte dekoda audiofil: {e}") from e

        # Resampla om nödvändigt
        if sr != TARGET_SR:
            audio_array = self._resample(audio_array, sr, TARGET_SR)
            sr = TARGET_SR

        # Konvertera till mono
        if audio_array.ndim > 1 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1)

        # Normalisera amplitud (peak normalisering, undviker klippning)
        peak = np.abs(audio_array).max()
        if peak > 0:
            audio_array = audio_array / peak * 0.95

        # Beräkna metadata
        duration_sec = len(audio_array) / sr
        metadata = {
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sr,
            "channels": 1,
            "num_samples": len(audio_array),
            "original_format": ext,
        }

        # Exportera till WAV-bytes
        wav_bytes = self._to_wav_bytes(audio_array, sr)

        log.info("ingest_complete", duration_sec=duration_sec, sample_rate=sr)
        return wav_bytes, metadata

    def _decode(self, audio_bytes: bytes, ext: str) -> Tuple[np.ndarray, int, dict]:
        """Dekoda audio till numpy-array. Försöker soundfile först, pydub som fallback."""
        buf = io.BytesIO(audio_bytes)

        # soundfile — hanterar WAV, FLAC, AIFF direkt
        if ext in (".wav", ".flac", ".aiff", ".aif"):
            try:
                arr, sr = sf.read(buf, dtype="float32", always_2d=True)
                return arr, sr, {}
            except Exception as e:
                log.warning("soundfile_failed", ext=ext, error=str(e))

        # pydub — hanterar MP3, OGG och övriga
        try:
            from pydub import AudioSegment
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            segment = AudioSegment.from_file(tmp_path)
            os.unlink(tmp_path)

            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            samples /= (2 ** (segment.sample_width * 8 - 1))  # Normalisera till [-1, 1]

            if segment.channels == 2:
                samples = samples.reshape(-1, 2)
            else:
                samples = samples.reshape(-1, 1)

            return samples, segment.frame_rate, {}

        except Exception as e:
            raise RuntimeError(f"pydub-dekodning misslyckades: {e}") from e

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resampla med librosa."""
        try:
            import librosa
            mono = audio.mean(axis=1) if audio.ndim > 1 else audio
            resampled = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr)
            return resampled
        except Exception as e:
            log.warning("resample_failed", error=str(e))
            return audio.mean(axis=1) if audio.ndim > 1 else audio

    def _to_wav_bytes(self, audio: np.ndarray, sr: int) -> bytes:
        """Exportera numpy-array som WAV-bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
        buf.seek(0)
        return buf.read()
