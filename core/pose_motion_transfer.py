"""
AI Pose & Motion Transfer Module.

Strategy (in priority order):
  1. Viggle API  — POST /api/render with ref_image + driving_video (needs VIGGLE_API_KEY)
  2. Replicate   — runs MimicMotion on GPU cloud (needs REPLICATE_API_TOKEN)
  3. Hard fail   — raises informative error, never silently falls back to a static video
"""
import os
import time
import subprocess
import requests
import random
from pathlib import Path
from typing import Optional

from config.settings import (
    OUTPUT_DIR, ASSETS_DIR, REPLICATE_API_TOKEN,
    TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS,
)

MOTION_REF_DIR = ASSETS_DIR / "motion_references"
MOTION_REF_DIR.mkdir(exist_ok=True)

VIGGLE_API_KEY = os.environ.get("VIGGLE_API_KEY", "")
VIGGLE_BASE    = "https://apis.viggle.ai"


# ---------------------------------------------------------------------------
# Strategy 1: Viggle API  (image + driving video → dancing video)
# ---------------------------------------------------------------------------

def _run_viggle(
    model_image_path: Path,
    reference_video_path: Path,
    output_path: Path,
) -> bool:
    """
    Submit a render job to the Viggle API and poll until complete.
    Returns True and writes the result to output_path on success.
    """
    key = VIGGLE_API_KEY
    if not key:
        print("ℹ️  [Viggle] No VIGGLE_API_KEY set — skipping.")
        return False

    headers = {"Authorization": f"Bearer {key}"}

    print("🚀 [Viggle] Submitting dance transfer job...")
    try:
        with open(model_image_path, "rb") as img, \
             open(reference_video_path, "rb") as vid:
            resp = requests.post(
                f"{VIGGLE_BASE}/api/render",
                headers=headers,
                files={
                    "ref_image":     ("model.jpg",  img, "image/jpeg"),
                    "driving_video": ("dance.mp4",  vid, "video/mp4"),
                },
                timeout=60,
            )

        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("job_id") or job.get("id")
        if not job_id:
            print(f"❌ [Viggle] Unexpected response (no job_id): {job}")
            return False

        print(f"✅ [Viggle] Job submitted: {job_id} — polling for result...")

        # Poll up to 10 minutes
        for attempt in range(120):
            time.sleep(5)
            status_resp = requests.get(
                f"{VIGGLE_BASE}/api/render/{job_id}",
                headers=headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            status = status_resp.json()
            state = status.get("status", "")
            print(f"   [{attempt+1}] status: {state}")

            if state == "complete":
                cdn_url = status.get("cdn_url") or status.get("output_url") or status.get("url")
                if not cdn_url:
                    print(f"❌ [Viggle] Complete but no cdn_url in response: {status}")
                    return False
                print(f"🎬 [Viggle] Downloading result from {cdn_url}...")
                video_resp = requests.get(cdn_url, timeout=120)
                video_resp.raise_for_status()
                output_path.write_bytes(video_resp.content)
                print(f"✅ [Viggle] Dance video saved: {output_path}")
                return True

            if state in ("failed", "error", "cancelled"):
                print(f"❌ [Viggle] Job failed with status '{state}': {status}")
                return False

        print("❌ [Viggle] Timed out after 10 minutes waiting for render.")
        return False

    except requests.HTTPError as e:
        print(f"❌ [Viggle] HTTP error: {e.response.status_code} — {e.response.text[:300]}")
        return False
    except Exception as e:
        print(f"❌ [Viggle] Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# Strategy 2: Replicate MimicMotion
# ---------------------------------------------------------------------------

def _run_replicate(
    model_image_path: Path,
    reference_video_path: Path,
    output_path: Path,
) -> bool:
    token = REPLICATE_API_TOKEN or os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        print("ℹ️  [Replicate] No REPLICATE_API_TOKEN set — skipping.")
        return False

    try:
        import replicate
        print("🚀 [Replicate] Submitting MimicMotion job...")
        with open(model_image_path, "rb") as img, \
             open(reference_video_path, "rb") as vid:
            output = replicate.run(
                "camenduru/mimic-motion:latest",
                input={"image": img, "video": vid,
                       "width": TARGET_WIDTH, "height": TARGET_HEIGHT},
            )
        url = list(output)[0] if hasattr(output, "__iter__") and not isinstance(output, str) else str(output)
        print(f"✅ [Replicate] Downloading from {url}...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        output_path.write_bytes(r.content)
        print(f"✅ [Replicate] Saved: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️  [Replicate] Failed: {e}")
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
    Transfer dance/body motion from reference_video_path onto model_image_path.

    Priority:
      1. Viggle API  (if VIGGLE_API_KEY is set)
      2. Replicate   (if REPLICATE_API_TOKEN is set)
      3. Hard fail with clear message
    """
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

    print(f"💃 [Motion Transfer] '{model_image_path.name}' + '{reference_video_path.name}' → dance video")

    output_path = OUTPUT_DIR / f"pose_transfer_{int(time.time())}.mp4"

    # 1. Viggle
    if use_cloud_gpu and _run_viggle(model_image_path, reference_video_path, output_path):
        return output_path

    # 2. Replicate
    if use_cloud_gpu and _run_replicate(model_image_path, reference_video_path, output_path):
        return output_path

    print("❌ [Motion Transfer] No API key available.")
    print("   → Set VIGGLE_API_KEY as a GitHub Actions secret and re-run the workflow.")
    return None
