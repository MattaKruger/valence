import argparse
import multiprocessing
import os

from audio_features import AudioFeatureExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features from music files"
    )
    parser.add_argument("directory", help="Directory containing audio files to analyze")
    parser.add_argument(
        "--output",
        "-o",
        default="audio_features.csv",
        help="Output CSV file name (default: audio_features.csv)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Sample rate for processing (default: use file's rate)",
    )
    parser.add_argument(
        "--n_fft", type=int, default=2048, help="FFT window size (default: 2048)"
    )
    parser.add_argument(
        "--hop_length", type=int, default=512, help="Hop length (default: 512)"
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help=f"Number of processing threads (default: CPU count-1, "
        f"current default would be {max(1, multiprocessing.cpu_count() - 1)})",
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        return 1

    # Create and use the feature extractor
    extractor = AudioFeatureExtractor(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sr=args.sr,
        num_threads=args.threads,
    )

    # Analyze the directory
    result = extractor.analyze_directory(args.directory, args.output)

    if result is not None:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
