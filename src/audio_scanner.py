import os
import sys


def scan_for_music_files(directory_path):
    if not os.path.exists(directory_path):
        print(f"Error: Directory: '{directory_path}' does not exist")
        return []

    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory")

    music_files = []

    music_extensions = [".wav", ".mp3", ".flac"]

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in music_extensions:
                music_files.append(file_path)

    return music_files


def main():
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = input("Enter the directory path to scan for music files: ")

    directory_path = os.path.abspath(directory_path)

    print(f"Scanning '{directory_path}' for music files...")

    music_files = scan_for_music_files(directory_path)

    if music_files:
        print(f"\nFound {len(music_files)} music file(s):")
        for file in music_files:
            print(f" - {file}")
    else:
        print("No music files found in the specified directory.")


if __name__ == "__main__":
    main()
