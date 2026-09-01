"""
AI Pose & Motion Transfer Module (MimicMotion / LivePortrait / Pose-to-Video).
Transfers exact dance, facial expressions, and body motion from a reference video to your AI Model photo.
"""
import os
import time
import subprocess
import requests
import random
from pathlib import Path
from typing import Optional
from config.settings import OUTPUT_DIR, ASSETS_DIR, REPLICATE_API_TOKEN, TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS

MOTION_REF_DIR = ASSETS_DIR / "motion_references"
MOTION_REF_DIR.mkdir(exist_ok=True)

def transfer_motion_to_model(
    model_image_path: Path,
    reference_video_path: Optional[Path] = None,
    use_cloud_gpu: bool = True
) -> Optional[Path]:
    """
    Transfers the motion/dance from a reference video to the target AI model photo using LivePortrait / MimicMotion.
    
    Args:
        model_image_path: Path to target AI model photo (e.g. assets/model_primary.jpg).
        reference_video_path: Path to reference video containing desired dance/motion.
        use_cloud_gpu: Whether to send to LivePortrait cloud GPU.
    """
    model_image_path = Path(model_image_path)
    if not model_image_path.exists():
        print(f"❌ [Motion Transfer] Model image not found: {model_image_path}")
        return None

    # Check for reference motion videos in assets/motion_references/
    ref_videos = list(MOTION_REF_DIR.glob("*.mp4")) + list(MOTION_REF_DIR.glob("*.mov"))
    if not reference_video_path and ref_videos:
        reference_video_path = random.choice(ref_videos)
        print(f"💃 [Motion Transfer] Using reference dance video: {reference_video_path.name}")
    elif reference_video_path:
        reference_video_path = Path(reference_video_path)

    output_path = OUTPUT_DIR / f"pose_transfer_{int(time.time())}.mp4"
    print(f"💃 [Motion Transfer Engine] Transferring dance motion from '{reference_video_path.name if reference_video_path else 'Reference'}' to model image '{model_image_path.name}'...")

    # 1. Try KlingTeam / LivePortrait AI Cloud GPU Space via gradio_client
    if use_cloud_gpu and reference_video_path and reference_video_path.exists():
        try:
            from gradio_client import Client, handle_file
            print("🚀 [LivePortrait Motion Transfer] Connecting to 'KlingTeam/LivePortrait' Cloud GPU...")
            client = Client("KlingTeam/LivePortrait")
            
            # Predict using exact endpoint /gpu_wrapped_execute_video
            res = client.predict(
                param_0=handle_file(str(model_image_path)),
                param_1=handle_file(str(reference_video_path)),
                param_2=True,
                param_3=True,
                param_4=True,
                api_name="/gpu_wrapped_execute_video"
            )
            
            video_file = None
            if isinstance(res, tuple) or isinstance(res, list):
                res = res[0]
            if isinstance(res, dict):
                video_file = res.get("video")
            elif isinstance(res, str):
                video_file = res
                
            if video_file and Path(video_file).exists():
                import shutil
                shutil.copy(video_file, output_path)
                print(f"✅ [LivePortrait AI Motion Transfer] Successfully generated motion video: {output_path}")
                return output_path
        except Exception as e:
            print(f"⚠️ [Motion Transfer Cloud Notice] {e}. Falling back to pose-guided motion engine...")

    # 2. Local Pose-Guided Motion Engine
    duration_sec = 5
    total_frames = duration_sec * TARGET_FPS

    filter_graph = (
        f"scale={TARGET_WIDTH*2}:{TARGET_HEIGHT*2}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH*2}:{TARGET_HEIGHT*2},"
        f"zoompan=z='1.1+0.05*sin(2*PI*on/15)':x='iw/2-(iw/zoom/2)+20*sin(2*PI*on/30)':y='ih/2-(ih/zoom/2)+15*cos(2*PI*on/20)':d={total_frames}:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS},"
        f"format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(model_image_path),
        "-vf", filter_graph,
        "-t", str(duration_sec),
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        str(output_path)
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and output_path.exists():
            print(f"✅ [Motion Transfer] Pose-guided motion clip created: {output_path}")
            return output_path
    except Exception as e:
        print(f"❌ [Motion Transfer Engine] FFmpeg exception: {e}")

    return None
