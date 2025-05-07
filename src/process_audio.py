import os
import time

import librosa
import librosa.display  # For visualization (optional)
import matplotlib.pyplot as plt  # For visualization (optional)
import numpy as np
import pandas as pd


def extract_features(audio_path):
    """
    Extracts a set of features from an audio file relevant for electronic music.

    Args:
        audio_path (str): Path to the .wav audio file.

    Returns:
        dict: A dictionary containing extracted features and their summary statistics,
              or None if loading fails.
        float: Duration of the track in seconds.
    """
    features = {}
    duration = 0.0

    try:
        print(f"Processing: {os.path.basename(audio_path)}...")
        start_time = time.time()

        # 1. Load Audio (consider using a lower sample rate for speed if acceptable)
        # sr=None preserves the original sample rate.
        # sr=22050 is common for feature extraction and faster processing.
        y, sr = librosa.load(audio_path, sr=None, mono=True)  # Load mono
        duration = librosa.get_duration(y=y, sr=sr)
        features["duration"] = duration

        # --- Rhythmic Features ---

        # 2. Tempo (BPM) - Crucial for Techno
        # Use an aggregate statistic as tempo might slightly fluctuate
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo)  # Ensure it's a standard float
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # features['beat_times'] = beat_times # Store actual beat times (can be large)
        features["num_beats"] = len(beat_times)

        # 3. Onset Detection (Detecting kicks, snares, hi-hats, etc.)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        # features['onset_times'] = onset_times # Store onset times (can be large)
        features["num_onsets"] = len(onset_times)
        features["onset_rate"] = len(onset_times) / duration if duration > 0 else 0

        # --- Spectral Features (Timbre/Texture) ---

        # Use a consistent frame size and hop length for time-series features
        n_fft = 2048
        hop_length = 512

        # 4. Spectral Centroid (Relates to "brightness")
        cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_centroid_mean"] = float(np.mean(cent))
        features["spectral_centroid_std"] = float(np.std(cent))

        # 5. Spectral Bandwidth (Measures width of the spectral shape)
        bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_bandwidth_mean"] = float(np.mean(bw))
        features["spectral_bandwidth_std"] = float(np.std(bw))

        # 6. Spectral Contrast (Difference between peaks and valleys in spectrum)
        # Useful for distinguishing tonal vs noisy parts
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))

        # 7. Spectral Rolloff (Frequency below which a percentage of energy lies)
        # Can indicate overall spectral shape/skewness
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        )[0]  # 85% is common
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # --- Energy/Loudness Features ---

        # 8. RMS Energy (Root Mean Square - related to perceived loudness)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))  # Dynamic range indicator

        # --- Harmonic/Pitch Features (Less central for techno, but can be useful) ---

        # 9. Chroma Features (Captures intensity of 12 pitch classes)
        # Useful for basslines or simple melodic motifs
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features["chroma_mean"] = [
            float(x) for x in np.mean(chroma, axis=1)
        ]  # Mean for each pitch class
        features["chroma_std"] = [
            float(x) for x in np.std(chroma, axis=1)
        ]  # Std dev for each pitch class

        # 10. Zero-Crossing Rate (Related to noisiness / percussiveness)
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=n_fft, hop_length=hop_length
        )[0]
        features["zero_crossing_rate_mean"] = float(np.mean(zcr))
        features["zero_crossing_rate_std"] = float(np.std(zcr))

        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13
        )  # Start with 13
        features["mfcc_mean"] = [float(x) for x in np.mean(mfccs, axis=1)]
        features["mfcc_std"] = [float(x) for x in np.std(mfccs, axis=1)]

        # Tempogram
        hop_length_tempo = 512  # Use a suitable hop length for tempogram
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length_tempo)
        tempogram = librosa.feature.tempogram(
            onset_envelope=oenv, sr=sr, hop_length=hop_length_tempo
        )
        # Aggregate tempogram - e.g., mean across time for each tempo bin, or find peak tempo
        features["tempogram_mean_vector"] = [
            float(x) for x in np.mean(tempogram, axis=1)
        ]  # Example aggregation

        # Tonnetz
        y_harmonic = librosa.effects.harmonic(y)  # Often better on harmonic part
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        features["tonnetz_mean"] = [float(x) for x in np.mean(tonnetz, axis=1)]
        features["tonnetz_std"] = [float(x) for x in np.std(tonnetz, axis=1)]
        end_time = time.time()
        print(f"   -> Extracted features in {end_time - start_time:.2f} seconds.")

        return features, duration

    except FileNotFoundError:
        print(f"Error: File not found at {audio_path}")
        return None, 0.0
    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        # Consider logging the error in more detail if needed
        # import traceback
        # traceback.print_exc()
        return None, 0.0


def analyze_directory(directory_path, output_csv="techno_features.csv"):
    """
    Analyzes all .wav files in a directory and saves features to a CSV.

    Args:
        directory_path (str): Path to the directory containing .wav files.
        output_csv (str): Name of the CSV file to save results.
    """
    all_features = []
    total_duration = 0.0

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return

    start_analysis_time = time.time()
    file_count = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            features, duration = extract_features(file_path)
            if features:
                features["filename"] = filename  # Add filename for reference
                all_features.append(features)
                total_duration += duration
                file_count += 1

    if not all_features:
        print("No valid .wav files found or processed in the directory.")
        return

    # Convert list of dictionaries to Pandas DataFrame
    df = pd.DataFrame(all_features)

    # Optional: Reorder columns for better readability
    cols = ["filename", "duration", "tempo", "num_beats"] + [
        col
        for col in df.columns
        if col not in ["filename", "duration", "tempo", "num_beats"]
    ]
    df = df[cols]

    # Save to CSV
    try:
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\nAnalysis complete for {file_count} tracks.")
        print(
            f"Total audio duration processed: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}"
        )
        print(f"Total analysis time: {time.time() - start_analysis_time:.2f} seconds.")
        print(f"Features saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving features to CSV: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to your directory containing techno .wav files
    techno_dir = (
        r"./data/radiance/"  # Use raw string (r"...") or forward slashes for paths
    )

    analyze_directory(techno_dir, output_csv="my_techno_analysis.csv")

    # --- Optional: Visualize a feature for one track ---
    # Load the generated CSV to easily select a file
    try:
        results_df = pd.read_csv("my_techno_analysis.csv")
        if not results_df.empty:
            sample_file = os.path.join(
                techno_dir, results_df.iloc[0]["filename"]
            )  # Get first file from results
            print(f"\nVisualizing RMS for: {results_df.iloc[0]['filename']}")

            y, sr = librosa.load(sample_file, sr=None, mono=True)
            n_fft = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

            plt.figure(figsize=(12, 4))
            plt.plot(times, rms)
            plt.title(f"RMS Energy - {os.path.basename(sample_file)}")
            plt.xlabel("Time (s)")
            plt.ylabel("RMS Amplitude")
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("\nCSV file is empty, skipping visualization.")

    except FileNotFoundError:
        print("\nCSV file 'my_techno_analysis.csv' not found. Run analysis first.")
    except ImportError:
        print("\nInstall matplotlib (`pip install matplotlib`) to visualize features.")
    except Exception as e:
        print(f"\nCould not visualize feature: {e}")
