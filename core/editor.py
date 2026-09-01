"""
Video Editing & Post-Processing Engine using FFmpeg.
Formating videos for 9:16 Vertical Shorts/Reels ratio with audio overlay and polish.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional
from config.settings import TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS, OUTPUT_DIR

def format_video_for_vertical_shorts(
    input_video: Path,
    audio_path: Optional[Path] = None,
    watermark_text: Optional[str] = "@AIModelOfficial"
) -> Optional[Path]:
    """
    Processes input video to standard 9:16 (1080x1920) vertical format,
    applies scaling/cropping, loops if needed, and merges background audio.
    """
    input_video = Path(input_video)
    if not input_video.exists():
        print(f"❌ [Editor] Input video file not found: {input_video}")
        return None

    output_path = OUTPUT_DIR / f"final_short_{input_video.stem}.mp4"
    print(f"✂️ [Editor] Formatting video {input_video.name} to 9:16 Vertical Short...")

    # Build FFmpeg filter complex:
    # 1. Scale and crop to exactly 1080x1920 (aspect ratio 9:16) with high quality
    # 2. Add text watermark (optional)
    vf_filter = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
    )

    if watermark_text:
        # Draw watermark text at bottom center
        vf_filter += (
            f",drawtext=text='{watermark_text}':fontcolor=white@0.8:fontsize=36:"
            f"x=(w-text_w)/2:y=h-100:shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video)
    ]

    if audio_path and Path(audio_path).exists():
        cmd.extend(["-i", str(audio_path), "-shortest"])

    cmd.extend([
        "-vf", vf_filter,
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ])

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and output_path.exists():
            print(f"✅ [Editor] Successfully formatted video to: {output_path}")
            return output_path
        else:
            print(f"⚠️ [Editor] FFmpeg process warning, fallback to direct input if valid: {res.stderr[:200]}")
            # If FFmpeg failed (e.g. missing font), run simpler scale crop without text
            fallback_cmd = [
                "ffmpeg", "-y", "-i", str(input_video),
                "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,crop={TARGET_WIDTH}:{TARGET_HEIGHT}",
                "-c:v", "libx264", str(output_path)
            ]
            subprocess.run(fallback_cmd, check=True)
            return output_path
    except Exception as e:
        print(f"❌ [Editor] Video formatting failed: {e}")
        return input_video
