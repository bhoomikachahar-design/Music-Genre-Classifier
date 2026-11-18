"""
Core Audio Processing Module for Music Genre Classifier & Explorer
Extracts GTZAN-compatible audio features, waveforms, and MFCC heatmaps.
"""

import io
import math
import numpy as np
import librosa
import soundfile as sf

STANDARD_SR = 22050
TARGET_DURATION = 30.0  # seconds
TARGET_LENGTH = 661794.0  # GTZAN standard length

class AudioProcessor:
    def __init__(self, target_sr: int = STANDARD_SR):
        self.target_sr = target_sr

    def load_audio(self, file_source, max_duration: float = TARGET_DURATION):
        """
        Loads audio from a file path or file-like bytes buffer.
        Converts to mono and resamples to target_sr (22,050 Hz).
        """
        try:
            if isinstance(file_source, (bytes, bytearray)):
                file_obj = io.BytesIO(file_source)
                y, sr = librosa.load(file_obj, sr=self.target_sr, mono=True, duration=max_duration)
            else:
                y, sr = librosa.load(file_source, sr=self.target_sr, mono=True, duration=max_duration)
        except Exception as e:
            raise ValueError(f"Failed to decode audio file. Please upload a valid audio format (WAV, MP3, OGG, FLAC). Error: {str(e)}")

        if y is None or len(y) == 0:
            raise ValueError("Audio signal is empty or corrupt.")

        # Minimum duration check: at least 0.8 seconds
        duration = len(y) / sr
        if duration < 0.8:
            raise ValueError(f"Audio is too short ({duration:.2f}s). Please provide a clip of at least 1 second.")

        # Peak amplitude normalization
        max_amp = np.max(np.abs(y))
        if max_amp > 1e-4:
            y = y / max_amp

        return y, sr

    def extract_gtzan_features(self, y: np.ndarray, sr: int):
        """
        Extracts the exact 58 features expected by the trained ML models.
        """
        feats = {}
        
        # Standard GTZAN duration length
        feats['length'] = TARGET_LENGTH

        # 1. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats['chroma_stft_mean'] = float(np.mean(chroma))
        feats['chroma_stft_var'] = float(np.var(chroma))

        # 2. RMS Energy
        rms = librosa.feature.rms(y=y)
        feats['rms_mean'] = float(np.mean(rms))
        feats['rms_var'] = float(np.var(rms))

        # 3. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats['spectral_centroid_mean'] = float(np.mean(cent))
        feats['spectral_centroid_var'] = float(np.var(cent))

        # 4. Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats['spectral_bandwidth_mean'] = float(np.mean(bw))
        feats['spectral_bandwidth_var'] = float(np.var(bw))

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feats['rolloff_mean'] = float(np.mean(rolloff))
        feats['rolloff_var'] = float(np.var(rolloff))

        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        feats['zero_crossing_rate_mean'] = float(np.mean(zcr))
        feats['zero_crossing_rate_var'] = float(np.var(zcr))

        # 7. Harmonic & Percussive Components
        harmony, perceptr = librosa.effects.hpss(y)
        feats['harmony_mean'] = float(np.mean(harmony))
        feats['harmony_var'] = float(np.var(harmony))
        feats['perceptr_mean'] = float(np.mean(perceptr))
        feats['perceptr_var'] = float(np.var(perceptr))

        # 8. Tempo
        tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo_val, '__len__'):
            tempo_val = tempo_val[0] if len(tempo_val) > 0 else 120.0
        feats['tempo'] = float(tempo_val)

        # 9. MFCCs (1 to 20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            feats[f'mfcc{i}_mean'] = float(np.mean(mfccs[i - 1]))
            feats[f'mfcc{i}_var'] = float(np.var(mfccs[i - 1]))

        return feats, mfccs

    def generate_visualizations(self, y: np.ndarray, sr: int, mfccs: np.ndarray, num_wave_points: int = 300):
        """
        Generates lightweight visualization payloads for the web frontend:
        - Downsampled waveform
        - Downsampled MFCC matrix for heatmap rendering
        - Downsampled Mel Spectrogram
        - Audio summary statistics
        """
        # 1. Waveform downsampling
        total_samples = len(y)
        duration = float(total_samples / sr)
        step = max(1, total_samples // num_wave_points)
        waveform_points = []
        for i in range(0, total_samples, step):
            chunk = y[i:i + step]
            if len(chunk) > 0:
                max_v = float(np.max(chunk))
                min_v = float(np.min(chunk))
                # take peak envelope
                peak = max_v if abs(max_v) > abs(min_v) else min_v
                waveform_points.append(round(peak, 3))
            if len(waveform_points) >= num_wave_points:
                break

        # 2. Downsample MFCC matrix for responsive heatmap (20 rows x ~60 columns)
        n_mfcc, n_frames = mfccs.shape
        target_cols = min(60, n_frames)
        col_step = max(1, n_frames // target_cols)
        heatmap_matrix = []
        for r in range(n_mfcc):
            row_data = []
            for c in range(0, n_frames, col_step):
                val = float(np.mean(mfccs[r, c:c + col_step]))
                row_data.append(round(val, 2))
                if len(row_data) >= target_cols:
                    break
            heatmap_matrix.append(row_data)

        # 3. Mel-Spectrogram matrix (24 bands x 50 time steps)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=24)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        s_bands, s_frames = mel_db.shape
        spec_cols = min(50, s_frames)
        spec_step = max(1, s_frames // spec_cols)
        spectrogram_matrix = []
        for r in range(s_bands):
            s_row = []
            for c in range(0, s_frames, spec_step):
                s_val = float(np.mean(mel_db[r, c:c + spec_step]))
                s_row.append(round(s_val, 2))
                if len(s_row) >= spec_cols:
                    break
            spectrogram_matrix.append(s_row)

        # 4. Summary metrics
        rms_val = float(np.sqrt(np.mean(y**2)))
        summary = {
            "duration": round(duration, 2),
            "sample_rate": sr,
            "samples_count": total_samples,
            "rms_energy_db": round(float(20 * math.log10(max(rms_val, 1e-5))), 1),
            "peak_amplitude": round(float(np.max(np.abs(y))), 3),
            "estimated_tempo_bpm": round(float(self.extract_tempo(y, sr)), 1)
        }

        return {
            "waveform": waveform_points,
            "mfcc_heatmap": {
                "rows": 20,
                "cols": len(heatmap_matrix[0]) if heatmap_matrix else 0,
                "matrix": heatmap_matrix,
                "labels": [f"MFCC {i}" for i in range(1, 21)]
            },
            "spectrogram": {
                "bands": 24,
                "cols": len(spectrogram_matrix[0]) if spectrogram_matrix else 0,
                "matrix": spectrogram_matrix
            },
            "summary": summary
        }

    def extract_tempo(self, y: np.ndarray, sr: int) -> float:
        try:
            tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(tempo_val, '__len__'):
                return float(tempo_val[0]) if len(tempo_val) > 0 else 120.0
            return float(tempo_val)
        except Exception:
            return 120.0
