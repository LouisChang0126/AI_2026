import subprocess
import sys
import os

def download_audio(keyword, count=10, audio_format='mp3'):
    try:
        # Create a folder with the same name as the keyword
        save_dir = keyword.replace(" ", "_")
        os.makedirs(save_dir, exist_ok=True)
        
        # Search for YouTube video IDs
        search_query = f"ytsearch{count}:{keyword}"
        result = subprocess.run(
            ["yt-dlp", search_query, "--print", "%(id)s"],
            capture_output=True, text=True, check=True
        )
        
        video_ids = result.stdout.strip().split('\n')
        
        if not video_ids:
            print("No related videos found.")
            return
        
        print(f"Found {len(video_ids)} videos, starting audio download...")
        
        # Download audio
        for video_id in video_ids:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"Downloading: {video_url}")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", audio_format,
                "-o", os.path.join(save_dir, "%(title)s.%(ext)s"),
                video_url
            ])
        
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print("Error:", e, file=sys.stderr)

if __name__ == "__main__":
    download_audio("Drum Solo")
    download_audio("Piano Solo")
    download_audio("Violin Solo")
    download_audio("Acoustic Guitar Solo")
    download_audio("Electric Guitar Solo")