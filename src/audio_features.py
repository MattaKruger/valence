import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd


class AudioFeatureExtractor:
    """
    A class for extracting audio features from music files, with a focus on electronic music.
    Supports parallel processing of multiple files.
    """

    def __init__(self, n_fft=2048, hop_length=512, sr=None, num_threads=None):
        """
        Initialize the feature extractor with specified parameters.

        Args:
            n_fft (int): FFT window size
            hop_length (int): Number of samples between frames
            sr (int): Sample rate for audio processing (None uses the file's original rate)
            num_threads (int): Number of worker threads to use (None uses CPU count)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.num_threads = num_threads or max(1, multiprocessing.cpu_count() - 1)

    def _process_file_worker(self, file_path):
        """Worker function for parallel processing"""
        return self.extract_features(file_path)

    def detect_key(self, y, sr):
        """
        Detects the musical key of an audio segment.

        Args:
            y (numpy.ndarray): Audio time series
            str (int): Sampling rate

        Returns:
            str: Detected musical key (e.g., "C major", "A minor")
        """

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        major_templates = []
        minor_templates = []

        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        for i in range(12):
            major_templates.append(np.roll(major_profile, i))
            minor_templates.append(np.roll(minor_profile, i))

        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        scores = []

        # Test correlation with major keys
        for i, template in enumerate(major_templates):
            scores.append(
                (key_names[i] + " major", np.corrcoef(chroma_avg, template)[0, 1])
            )

        # Test correlation with minor keys
        for i, template in enumerate(minor_templates):
            scores.append(
                (key_names[i] + " minor", np.corrcoef(chroma_avg, template)[0, 1])
            )

        # Sort by correlation score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return the key with highest correlation
        return scores[0][0], scores[0][1]  # Returns key name and confidence score

    def extract_features(self, audio_path):
        """
        Extracts a set of features from an audio file relevant for electronic music.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            dict: A dictionary containing extracted features and their summary statistics,
                  or None if loading fails.
            float: Duration of the track in seconds.
        """
        features = {}
        duration = 0.0

        try:
            start_time = time.time()

            # Load Audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            features["duration"] = duration

            # Key Detection
            key_name, key_confidence = self.detect_key(y, sr)
            features["key"] = key_name
            features["key_confidence"] = float(key_confidence)

            # --- Rhythmic Features ---

            # Tempo (BPM)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            features["num_beats"] = len(beat_times)

            # Onset Detection
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            features["num_onsets"] = len(onset_times)
            features["onset_rate"] = len(onset_times) / duration if duration > 0 else 0

            # --- Spectral Features ---

            # Spectral Centroid
            cent = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            features["spectral_centroid_mean"] = float(np.mean(cent))
            features["spectral_centroid_std"] = float(np.std(cent))

            # Spectral Bandwidth
            bw = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            features["spectral_bandwidth_mean"] = float(np.mean(bw))
            features["spectral_bandwidth_std"] = float(np.std(bw))

            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["spectral_contrast_mean"] = float(np.mean(contrast))
            features["spectral_contrast_std"] = float(np.std(contrast))

            # Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=y,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                roll_percent=0.85,
            )[0]
            features["spectral_rolloff_mean"] = float(np.mean(rolloff))
            features["spectral_rolloff_std"] = float(np.std(rolloff))

            # --- Energy/Loudness Features ---

            # RMS Energy
            rms = librosa.feature.rms(
                y=y, frame_length=self.n_fft, hop_length=self.hop_length
            )[0]
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            # --- Harmonic/Pitch Features ---

            # Chroma Features
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["chroma_mean"] = [float(x) for x in np.mean(chroma, axis=1)]
            features["chroma_std"] = [float(x) for x in np.std(chroma, axis=1)]

            # Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(
                y, frame_length=self.n_fft, hop_length=self.hop_length
            )[0]
            features["zero_crossing_rate_mean"] = float(np.mean(zcr))
            features["zero_crossing_rate_std"] = float(np.std(zcr))

            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=13
            )
            features["mfcc_mean"] = [float(x) for x in np.mean(mfccs, axis=1)]
            features["mfcc_std"] = [float(x) for x in np.std(mfccs, axis=1)]

            # Tempogram
            hop_length_tempo = 512
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length_tempo)
            tempogram = librosa.feature.tempogram(
                onset_envelope=oenv, sr=sr, hop_length=hop_length_tempo
            )
            features["tempogram_mean_vector"] = [
                float(x) for x in np.mean(tempogram, axis=1)
            ]

            # Tonnetz
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            features["tonnetz_mean"] = [float(x) for x in np.mean(tonnetz, axis=1)]
            features["tonnetz_std"] = [float(x) for x in np.std(tonnetz, axis=1)]

            end_time = time.time()
            features["filename"] = os.path.basename(audio_path)
            features["file_path"] = audio_path

            print(
                f"Processed: {os.path.basename(audio_path)} in {end_time - start_time:.2f} seconds."
            )
            return features, duration

        except FileNotFoundError:
            print(f"Error: File not found at {audio_path}")
            return None, 0.0
        except Exception as e:
            print(f"Error processing {os.path.basename(audio_path)}: {e}")
            return None, 0.0

    def analyze_directory(self, directory_path, output_csv="audio_features.csv"):
        """
        Analyzes all supported audio files in a directory and saves features to a CSV.
        Uses multithreading to process multiple files in parallel.

        Args:
            directory_path (str): Path to the directory containing audio files.
            output_csv (str): Name of the CSV file to save results.

        Returns:
            pd.DataFrame or None: DataFrame containing the extracted features, or None if error
        """
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return None

        # Get list of supported audio files
        supported_extensions = [".wav", ".mp3", ".flac"]
        audio_files = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                ext = os.path.splitext(file.lower())[1]
                if ext in supported_extensions:
                    audio_files.append(os.path.join(root, file))

        if not audio_files:
            print(f"No supported audio files found in {directory_path}")
            return None

        print(
            f"Found {len(audio_files)} audio files. Processing with {self.num_threads} threads..."
        )
        start_analysis_time = time.time()

        # Process files in parallel using ThreadPoolExecutor
        all_features = []
        total_duration = 0.0
        file_count = 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_file_worker, f): f for f in audio_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    features, duration = future.result()
                    if features:
                        all_features.append(features)
                        total_duration += duration
                        file_count += 1
                except Exception as e:
                    print(f"Error processing {os.path.basename(file_path)}: {e}")

        if not all_features:
            print("No valid audio files processed in the directory.")
            return None

        # Convert to DataFrame
        try:
            df = pd.DataFrame(all_features)

            # Organize columns
            essential_cols = ["filename", "file_path", "duration", "tempo", "num_beats"]
            remaining_cols = [col for col in df.columns if col not in essential_cols]
            df = df[essential_cols + remaining_cols]

            # Save to CSV
            df.to_csv(output_csv, index=False, encoding="utf-8")

            total_time = time.time() - start_analysis_time
            print(f"\nAnalysis complete for {file_count} tracks.")
            print(
                f"Total audio duration processed: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}"
            )
            print(f"Total analysis time: {total_time:.2f} seconds")
            print(
                f"Average processing time per file: {total_time / file_count:.2f} seconds"
            )
            print(f"Features saved to: {output_csv}")
            return df

        except Exception as e:
            print(f"Error creating DataFrame or saving to CSV: {e}")
            return None
