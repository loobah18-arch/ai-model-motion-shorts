"""
100% FREE Motion Video Generators (Phone-Optimized & $0 Cost).

Provides two free methods:
1. Free Cloud AI Motion: Uses Hugging Face Spaces (ZeroGPU) via gradio_client.
2. Free On-Device Dynamic Motion: Uses FFmpeg cinematic pan/zoom & depth motion directly on phone.
"""
import os
import time
import subprocess
import random
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from config.settings import TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS, OUTPUT_DIR

CATALOG_JSON = OUTPUT_DIR / "catalog.json"
CATALOG_MD = OUTPUT_DIR / "CATALOG.md"

def log_video_to_catalog(video_info: Dict[str, Any]):
    """Logs generated video details to local catalog for review in Dev Mode."""
    catalog = []
    if CATALOG_JSON.exists():
        try:
            with open(CATALOG_JSON, "r") as f:
                catalog = json.load(f)
        except Exception:
            catalog = []
            
    catalog.append(video_info)
    
    with open(CATALOG_JSON, "w") as f:
        json.dump(catalog, f, indent=2)

    # Write human-readable markdown catalog
    with open(CATALOG_MD, "w") as f:
        f.write("# 🎥 AI Model Motion Videos - Local Dev Catalog\n\n")
        f.write("| Timestamp | Motion Preset | File Name | Size | Resolution |\n")
        f.write("|---|---|---|---|---|\n")
        for item in reversed(catalog):
            f.write(f"| {item.get('timestamp')} | `{item.get('motion_type')}` | `{item.get('file_name')}` | {item.get('file_size_mb')} MB | {item.get('resolution')} |\n")


def create_free_ffmpeg_motion(
    image_path: Path,
    motion_type: str = "zoom_in",
    duration_sec: int = 5
) -> Optional[Path]:
    """
    Creates a 100% free cinematic motion video from an AI model image directly on phone.
    No API keys, no internet required, $0 cost!
    
    Motion types:
    - zoom_in: Smooth 60fps cinematic zoom into AI model face/details
    - pan_down: Slow elegant camera pan down the fashion outfit
    - pan_up: Fashion runway slow motion pan up
    - subtle_orbit: Gentle breathing & camera push effect
    - parallax_shift: Dynamic lighting & micro camera shift
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ [Free Motion] Image file not found: {image_path}")
        return None

    output_path = OUTPUT_DIR / f"dev_motion_{motion_type}_{int(time.time())}.mp4"
    total_frames = duration_sec * TARGET_FPS

    print(f"🎬 [Dev Motion Engine] Generating '{motion_type}' motion clip from {image_path.name}...")

    # Define FFmpeg zoompan filter expressions for smooth 60fps/30fps motion
    if motion_type == "zoom_in":
        zoom_expr = "min(zoom+0.0015,1.25)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif motion_type == "pan_down":
        zoom_expr = "1.15"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "y+0.6"
    elif motion_type == "pan_up":
        zoom_expr = "1.15"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"max(0, ih-(on*{1.2}))"
    elif motion_type == "parallax_shift":
        zoom_expr = "1.1"
        x_expr = f"iw/2-(iw/zoom/2)+15*sin(2*PI*on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        # Subtle breathing push-in
        zoom_expr = f"1.0+0.08*sin(2*PI*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    filter_graph = (
        f"scale={TARGET_WIDTH*2}:{TARGET_HEIGHT*2}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH*2}:{TARGET_HEIGHT*2},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS},"
        f"format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", filter_graph,
        "-t", str(duration_sec),
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        str(output_path)
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and output_path.exists():
            size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
            print(f"✅ [Dev Motion Engine] Motion clip saved: {output_path.name} ({size_mb} MB)")
            
            log_video_to_catalog({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "motion_type": motion_type,
                "file_name": output_path.name,
                "file_path": str(output_path),
                "file_size_mb": size_mb,
                "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
            })
            return output_path
        else:
            print(f"❌ [Dev Motion Engine] FFmpeg error: {res.stderr[:200]}")
    except Exception as e:
        print(f"❌ [Dev Motion Engine] Exception: {e}")

    return None


def batch_generate_dev_motions(image_path: Path) -> List[Path]:
    """Generates a complete suite of motion presets for testing and reviewing."""
    motion_types = ["zoom_in", "pan_down", "pan_up", "subtle_orbit", "parallax_shift"]
    results = []
    print(f"📦 [Batch Dev Mode] Generating {len(motion_types)} motion presets for {image_path.name}...")
    for m in motion_types:
        out = create_free_ffmpeg_motion(image_path, motion_type=m)
        if out:
            results.append(out)
    return results
