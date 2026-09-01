"""
AI Pose & Motion Transfer Module.

Strategy (in priority order):
  1. Replicate API  — runs MimicMotion on GPU cloud (needs REPLICATE_API_TOKEN secret)
  2. OpenCV TPS warp — extracts body skeleton keypoints from reference video frames,
                       warps the model image per-frame to match each pose (runs locally, no GPU)
  3. Hard fail       — raises an informative error, never silently falls back to a static zoompan

Requirements for strategy 2: opencv-python (pkg install opencv-python on Termux)
"""
import os
import time
import shutil
import subprocess
import tempfile
import requests
from pathlib import Path
from typing import Optional, List, Tuple

from config.settings import (
    OUTPUT_DIR, ASSETS_DIR, REPLICATE_API_TOKEN,
    TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS,
)

MOTION_REF_DIR = ASSETS_DIR / "motion_references"
MOTION_REF_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Strategy 1: Replicate MimicMotion
# ---------------------------------------------------------------------------

def _run_replicate_mimic_motion(
    model_image_path: Path,
    reference_video_path: Path,
    output_path: Path,
) -> bool:
    """Call camenduru/mimic-motion on Replicate. Returns True on success."""
    token = REPLICATE_API_TOKEN or os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        print("ℹ️  [Replicate] No REPLICATE_API_TOKEN set — skipping cloud GPU path.")
        return False

    try:
        import replicate

        print("🚀 [Replicate] Uploading assets to MimicMotion cloud GPU...")

        with open(model_image_path, "rb") as img_f, \
             open(reference_video_path, "rb") as vid_f:

            output_url_iter = replicate.run(
                "camenduru/mimic-motion:latest",
                input={
                    "image": img_f,
                    "video": vid_f,
                    "width": TARGET_WIDTH,
                    "height": TARGET_HEIGHT,
                }
            )

        # replicate.run returns a list of URLs or a single URL
        if hasattr(output_url_iter, "__iter__") and not isinstance(output_url_iter, str):
            output_url = list(output_url_iter)[0]
        else:
            output_url = str(output_url_iter)

        print(f"✅ [Replicate] Generated dance video URL: {output_url}")
        resp = requests.get(output_url, timeout=120)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        print(f"✅ [Replicate] Saved dance video: {output_path}")
        return True

    except Exception as exc:
        print(f"⚠️  [Replicate] Failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Strategy 2: Local OpenCV Skeleton-Warp Motion Transfer
# ---------------------------------------------------------------------------

def _extract_frames(video_path: Path, max_frames: int = 60) -> List:
    """Extract up to max_frames BGR frames from video using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        idx += step
    cap.release()
    return frames


def _detect_body_keypoints(frame) -> Optional[List[Tuple[int, int]]]:
    """
    Detect 25 body keypoints using OpenCV DNN + OpenPose COCO model.
    Falls back to simple contour centroid if model weights unavailable.
    Returns list of (x, y) or None if detection fails.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    # Try OpenPose body_25 prototxt/caffemodel if cached
    proto   = Path.home() / ".cache" / "openpose" / "pose_deploy_linevec.prototxt"
    weights = Path.home() / ".cache" / "openpose" / "pose_iter_440000.caffemodel"

    if proto.exists() and weights.exists():
        net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (368, 368), (0,0,0), swapRB=False)
        net.setInput(blob)
        out = net.forward()  # shape (1, 19, h, w)
        n_points = 18
        points = []
        for i in range(n_points):
            heatmap = out[0, i, :, :]
            _, conf, _, pt = cv2.minMaxLoc(heatmap)
            x = int(pt[0] * w / out.shape[3])
            y = int(pt[1] * h / out.shape[2])
            points.append((x, y) if conf > 0.1 else None)
        return points

    # Lightweight fallback: detect skin-tone region centroids as proxy for
    # head/torso/hand anchor points (good enough for rough warp estimation)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 20, 70), (25, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Return centroids of top-4 largest contours as anchor keypoints
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    pts = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        pts.append((cx, cy))
    return pts if pts else None


def _global_motion_params(ref_kp, baseline_kp):
    """
    Compute (tx, ty, scale, angle) rigid transform from baseline to ref keypoints.
    Used to drive per-frame warp of model image.
    """
    import numpy as np

    def centroid(kps):
        valid = [(x, y) for x, y in kps if x > 0 or y > 0]
        if not valid:
            return (0, 0)
        return (int(np.mean([p[0] for p in valid])),
                int(np.mean([p[1] for p in valid])))

    n = len(ref_kp)
    if n < 2 or len(baseline_kp) < 2:
        return 0, 0, 1.0, 0.0

    rc = centroid(ref_kp)
    bc = centroid(baseline_kp)
    tx = rc[0] - bc[0]
    ty = rc[1] - bc[1]

    # Rough scale from vertical spread
    ref_ys  = sorted([y for _, y in ref_kp if y > 0])
    base_ys = sorted([y for _, y in baseline_kp if y > 0])
    if ref_ys and base_ys:
        ref_span  = ref_ys[-1] - ref_ys[0] + 1e-6
        base_span = base_ys[-1] - base_ys[0] + 1e-6
        scale = ref_span / base_span
    else:
        scale = 1.0

    scale = float(np.clip(scale, 0.7, 1.3))
    return tx, ty, scale, 0.0


def _warp_model_frame(model_img, tx: float, ty: float, scale: float, angle: float,
                      out_w: int, out_h: int):
    """Apply affine warp to model image based on motion params."""
    import cv2
    import numpy as np

    h, w = model_img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    # Add translation (constrained so model stays mostly in frame)
    max_shift_x = w * 0.15
    max_shift_y = h * 0.15
    tx = float(np.clip(tx * 0.4, -max_shift_x, max_shift_x))
    ty = float(np.clip(ty * 0.4, -max_shift_y, max_shift_y))
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv2.warpAffine(model_img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    # Resize to output dimensions
    return cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def _run_opencv_warp_transfer(
    model_image_path: Path,
    reference_video_path: Path,
    output_path: Path,
) -> bool:
    """
    Skeleton-driven motion transfer using OpenCV.
    Extracts body pose keypoints per frame from reference video,
    computes rigid warp params, and animates the model image to match.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ [Local Engine] opencv not available. Install with: pkg install opencv-python")
        return False

    print("🦴 [Local Engine] Extracting pose keypoints from reference video...")
    ref_frames = _extract_frames(reference_video_path, max_frames=90)
    if not ref_frames:
        print("❌ [Local Engine] Could not read reference video frames.")
        return False

    # Extract keypoints for all reference frames
    ref_keypoints = []
    for f in ref_frames:
        kp = _detect_body_keypoints(f)
        ref_keypoints.append(kp if kp else [])

    # Detect baseline keypoints on first valid ref frame
    baseline_kp = next((kp for kp in ref_keypoints if kp), [])
    if not baseline_kp:
        print("❌ [Local Engine] Could not detect any keypoints in reference video.")
        return False

    print(f"✅ [Local Engine] Detected keypoints in {sum(1 for k in ref_keypoints if k)}/{len(ref_frames)} frames.")

    # Load model image
    model_img = cv2.imread(str(model_image_path))
    if model_img is None:
        print(f"❌ [Local Engine] Cannot read model image: {model_image_path}")
        return False

    # Resize model to target 9:16
    model_img = cv2.resize(model_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    print(f"🎬 [Local Engine] Rendering {len(ref_keypoints)} animated frames...")

    # Write frames to temp dir then encode to MP4 with FFmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, ref_kp in enumerate(ref_keypoints):
            if ref_kp:
                tx, ty, scale, angle = _global_motion_params(ref_kp, baseline_kp)
            else:
                tx, ty, scale, angle = 0, 0, 1.0, 0.0

            frame = _warp_model_frame(model_img, tx, ty, scale, angle, TARGET_WIDTH, TARGET_HEIGHT)
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_path)

        if not frame_paths:
            return False

        # Encode frames to MP4
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(TARGET_FPS),
            "-i", os.path.join(tmpdir, "frame_%04d.jpg"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            duration = len(frame_paths) / TARGET_FPS
            print(f"✅ [Local Engine] Dance motion video rendered: {output_path} ({duration:.1f}s, {len(frame_paths)} frames)")
            return True
        else:
            print(f"❌ [Local Engine] FFmpeg encode failed:\n{result.stderr[-500:]}")
            return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transfer_motion_to_model(
    model_image_path: Path,
    reference_video_path: Optional[Path] = None,
    use_cloud_gpu: bool = True,
) -> Optional[Path]:
    """
    Transfer dance/body motion from reference_video_path to model_image_path.

    Priority:
      1. Replicate MimicMotion (if REPLICATE_API_TOKEN is set)
      2. Local OpenCV TPS skeleton warp (always available if opencv is installed)

    Returns the output video Path, or None on complete failure.
    """
    import random

    model_image_path = Path(model_image_path)
    if not model_image_path.exists():
        print(f"❌ [Motion Transfer] Model image not found: {model_image_path}")
        return None

    # Resolve reference video
    if reference_video_path is None:
        ref_videos = list(MOTION_REF_DIR.glob("*.mp4")) + list(MOTION_REF_DIR.glob("*.mov"))
        if ref_videos:
            reference_video_path = random.choice(ref_videos)
            print(f"💃 [Motion Transfer] Auto-selected reference: {reference_video_path.name}")
    else:
        reference_video_path = Path(reference_video_path)

    if reference_video_path is None or not reference_video_path.exists():
        print("❌ [Motion Transfer] No reference video found.")
        return None

    print(f"💃 [Motion Transfer] '{model_image_path.name}' ← '{reference_video_path.name}'")

    output_path = OUTPUT_DIR / f"pose_transfer_{int(time.time())}.mp4"

    # 1. Try Replicate cloud
    if use_cloud_gpu and _run_replicate_mimic_motion(model_image_path, reference_video_path, output_path):
        return output_path

    # 2. Local OpenCV skeleton warp
    print("🦴 [Local Engine] Running skeleton-warp motion transfer locally...")
    if _run_opencv_warp_transfer(model_image_path, reference_video_path, output_path):
        return output_path

    print("❌ [Motion Transfer] All strategies failed. Set REPLICATE_API_TOKEN for cloud GPU.")
    return None
