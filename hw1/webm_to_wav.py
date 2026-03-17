import os
import subprocess

clip_duration = 10  # Duration of each segment in seconds
num_clips = 10 # or 20 # Number of segments per video

def extract_audio_segments(input_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".webm"):
            video_path = os.path.join(input_folder, file)
            output_folder = os.path.join("dataset", input_folder)
            os.makedirs(output_folder, exist_ok=True)
            print(video_path)
            # Extract 20 segments of 10 seconds each from the webm video and convert to .wav
            filename = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Processing {filename}...")
            # Get the total duration of the video
            cmd_duration = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                total_duration = float(result.stdout.strip())
            except ValueError:
                print(f"Failed to get duration of {video_path}")
                return

            # Ensure the video is long enough
            if total_duration < clip_duration * num_clips:
                print(f"{video_path} is shorter than {clip_duration * num_clips} seconds, extracting available segments")

            # Extract audio segments
            for i in range(num_clips):
                start_time = i * clip_duration
                if start_time + clip_duration > total_duration:
                    break  # Avoid exceeding video duration

                output_wav = os.path.join(output_folder, f"{filename}_part{i+1}.wav")
                cmd_extract = [
                    "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
                    "-ac", "1", "-ss", str(start_time), "-t", str(clip_duration), output_wav, "-y"
                ]
                subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Extracted {output_wav}")

if __name__ == "__main__":
    extract_audio_segments("Drum_Solo")
    extract_audio_segments("Piano_Solo")
    extract_audio_segments("Violin_Solo")
    extract_audio_segments("Acoustic_Guitar_Solo")
    extract_audio_segments("Electric_Guitar_Solo")