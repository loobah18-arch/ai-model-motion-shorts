"""
AI Pose & Motion Transfer Module.

Strategy (in priority order):
  1. Kaggle T4 GPU — 30 hours/week FREE compute (needs KAGGLE_USERNAME & KAGGLE_KEY)
  2. Viggle API    — POST /api/render with ref_image + driving_video (needs VIGGLE_API_KEY)
  3. Replicate     — runs MimicMotion on GPU cloud (needs REPLICATE_API_TOKEN)
  4. Hard fail     — raises informative error, never silently falls back to a static video
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

VIGGLE_API_KEY  = os.environ.get("VIGGLE_API_KEY", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.environ.get("KAGGLE_KEY", "")
VIGGLE_BASE     = "https://apis.viggle.ai"


# ---------------------------------------------------------------------------
# Strategy 1: Kaggle T4 GPU Runner (30 Free Hours/Week)
# ---------------------------------------------------------------------------

def _run_kaggle(
    model_image_path: Path,
    reference_video_path: Path,
    output_path: Path,
) -> bool:
    """
    Triggers Kaggle T4 GPU kernel via Kaggle API, polls for completion,
    and downloads full-length 14-second dance transfer video.
    """
    user = KAGGLE_USERNAME or os.environ.get("KAGGLE_USERNAME", "")
    key  = KAGGLE_KEY or os.environ.get("KAGGLE_KEY", "")

    if not user or not key:
        print("ℹ️  [Kaggle GPU] KAGGLE_USERNAME or KAGGLE_KEY not set — skipping.")
        return False

    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"]      = key

    # Write ~/.kaggle/kaggle.json explicitly for CLI authentication
    import json
    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_config_dir.mkdir(exist_ok=True)
    config_file = kaggle_config_dir / "kaggle.json"
    config_file.write_text(json.dumps({"username": user, "key": key}))
    config_file.chmod(0o600)

    print(f"🚀 [Kaggle T4 GPU] Authenticated as user '{user}'. Pushing MimicMotion GPU kernel...")
    kaggle_dir = Path(__file__).resolve().parent.parent / "kaggle"

    try:
        # Push kernel using kaggle CLI
        push_cmd = ["kaggle", "kernels", "push", "-p", str(kaggle_dir)]
        res = subprocess.run(push_cmd, capture_output=True, text=True)
        print(f"   [Kaggle CLI] {res.stdout.strip()}")
        if res.returncode != 0 and "Kernel URL" not in res.stdout:
            print(f"⚠️  [Kaggle GPU] Push notice: {res.stderr.strip()[:200]}")

        kernel_slug = f"{user}/ai-model-mimic-motion-runner"
        print(f"✅ [Kaggle GPU] Kernel '{kernel_slug}' running on free T4 GPU — polling output...")

        # Poll status for up to 10 minutes
        for attempt in range(60):
            time.sleep(10)
            status_cmd = ["kaggle", "kernels", "status", kernel_slug]
            st = subprocess.run(status_cmd, capture_output=True, text=True)
            status_text = st.stdout.strip()
            print(f"   [{attempt+1}] Kaggle GPU status: {status_text}")

            if "complete" in status_text.lower():
                print("🎬 [Kaggle GPU] Execution complete! Downloading generated video output...")
                out_cmd = ["kaggle", "kernels", "output", kernel_slug, "-p", str(OUTPUT_DIR)]
                subprocess.run(out_cmd, check=True)

                downloaded = OUTPUT_DIR / "kaggle_dance_transfer.mp4"
                if downloaded.exists():
                    downloaded.rename(output_path)
                    print(f"✅ [Kaggle T4 GPU] Dance video successfully generated & saved: {output_path}")
                    return True

            if "error" in status_text.lower() or "failed" in status_text.lower():
                print(f"❌ [Kaggle GPU] Kernel run failed: {status_text}")
                break

        print("ℹ️  [Kaggle GPU] Kernel run timed out or output pending — trying API fallbacks...")
        return False

    except Exception as e:
        print(f"⚠️  [Kaggle GPU] Exception: {e}")
        return False


# ---------------------------------------------------------------------------
# Strategy 2: Viggle API  (image + driving video → dancing video)
# ---------------------------------------------------------------------------

def _run_viggle(
    model_image_path: Path,
    reference_video_path: Path,
    output_path: Path,
) -> bool:
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

        print("❌ [Viggle] Timed out waiting for render.")
        return False

    except requests.HTTPError as e:
        print(f"❌ [Viggle] HTTP error: {e.response.status_code} — {e.response.text[:300]}")
        return False
    except Exception as e:
        print(f"❌ [Viggle] Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# Strategy 3: Replicate zsxkib/mimic-motion
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
        client = replicate.Client(api_token=token)
        print("🚀 [Replicate] Submitting zsxkib/mimic-motion job...")

        with open(model_image_path, "rb") as img, \
             open(reference_video_path, "rb") as vid:
            output = client.run(
                "zsxkib/mimic-motion:b3edd455f68ec4ccf045da8732be7db837cb8832d1a2459ef057ddcd3ff87dea",
                input={
                    "appearance_image": img,
                    "motion_video":     vid,
                }
            )

        url = list(output)[0] if hasattr(output, "__iter__") and not isinstance(output, str) else str(output)
        print(f"✅ [Replicate] Render done — downloading from {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        output_path.write_bytes(r.content)
        print(f"✅ [Replicate] Saved dance video: {output_path}")
        return True

    except Exception as e:
        print(f"⚠️  [Replicate] Failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def transfer_motion_to_model(
    model_image_path: Path,
    reference_video_path: Optional[Path] = None,
    use_cloud_gpu: bool = True,
) -> Optional[Path]:
    """
    Transfer dance/body motion from reference_video_path onto model_image_path.

    Priority:
      1. Kaggle T4 GPU (30 Hours/Week FREE - Full 14s Video)
      2. Viggle API  (if VIGGLE_API_KEY set)
      3. Replicate   (if REPLICATE_API_TOKEN set)
    """
    model_image_path = Path(model_image_path)
    if not model_image_path.exists():
        print(f"❌ [Motion Transfer] Model image not found: {model_image_path}")
        return None

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

    # 1. Kaggle Free T4 GPU (Primary & Solo Test Mode)
    print("🧪 [Test Mode] Viggle & Replicate disabled. Testing Kaggle T4 GPU exclusively...")
    if use_cloud_gpu and _run_kaggle(model_image_path, reference_video_path, output_path):
        return output_path

    print("❌ [Kaggle Solo Test] Kaggle T4 GPU run failed.")
    return None
