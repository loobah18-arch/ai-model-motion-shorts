"""
AI Motion Video Generator using Replicate API (Image-to-Video & Text-to-Video).
"""
import os
import time
import requests
import random
from pathlib import Path
from typing import Optional, Dict, Any
from config.settings import REPLICATE_API_TOKEN, DEFAULT_MOTION_PROMPTS, OUTPUT_DIR

def generate_motion_video(
    image_path: Optional[Path] = None,
    prompt: Optional[str] = None,
    model_version: str = "minimax/video-01"
) -> Optional[Path]:
    """
    Generates a motion video clip from an image or prompt using Replicate API.
    
    Args:
        image_path: Path to input AI model image (optional for image-to-video).
        prompt: Motion prompt describing the desired animation/video action.
        model_version: Replicate video generation model identifier.
        
    Returns:
        Path to downloaded generated MP4 video file, or None if failed.
    """
    api_token = REPLICATE_API_TOKEN or os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("[Generator] ⚠️ REPLICATE_API_TOKEN not set in environment.")
        return None

    if not prompt:
        prompt = random.choice(DEFAULT_MOTION_PROMPTS)

    print(f"🎬 [Generator] Starting video generation with model: '{model_version}'")
    print(f"📝 Prompt: '{prompt}'")

    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    # Prepare prediction payload depending on model type
    input_payload = {
        "prompt": prompt,
        "prompt_optimizer": True
    }

    if image_path and Path(image_path).exists():
        print(f"🖼️ Using source model image: {image_path}")
        # Note: Replicate expects image URLs or base64 data URI
        import base64
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        input_payload["first_frame_image"] = f"data:image/jpeg;base64,{b64_img}"

    # Initiate prediction via Replicate API
    post_url = "https://api.replicate.com/v1/predictions"
    payload = {
        "version": model_version,
        "input": input_payload
    }

    try:
        # Create prediction using model deployment shortcut if model_version is full name
        if "/" in model_version:
            post_url = f"https://api.replicate.com/v1/models/{model_version}/predictions"
            payload = {"input": input_payload}

        response = requests.post(post_url, json=payload, headers=headers, timeout=30)
        if response.status_code not in (200, 201):
            print(f"❌ [Generator] API Request failed ({response.status_code}): {response.text}")
            return None

        prediction = response.json()
        prediction_id = prediction.get("id")
        get_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction_id}"

        print(f"⏳ [Generator] Polling generation progress (ID: {prediction_id})...")
        
        # Poll prediction status until succeeded or failed
        for _ in range(60): # up to 5 minutes timeout
            time.sleep(5)
            status_resp = requests.get(get_url, headers=headers, timeout=15)
            if status_resp.status_code != 200:
                continue
            
            data = status_resp.json()
            status = data.get("status")
            print(f"   Status: {status}")
            
            if status == "succeeded":
                output_url = data.get("output")
                if isinstance(output_url, list):
                    output_url = output_url[0]
                
                if not output_url:
                    print("❌ [Generator] Succeeded but output URL empty.")
                    return None
                    
                # Download video file
                video_resp = requests.get(output_url, timeout=60)
                output_file = OUTPUT_DIR / f"generated_motion_{int(time.time())}.mp4"
                with open(output_file, "wb") as f:
                    f.write(video_resp.content)
                
                print(f"✅ [Generator] Video successfully generated and saved to: {output_file}")
                return output_file
                
            elif status in ("failed", "canceled"):
                print(f"❌ [Generator] Video generation {status}: {data.get('error')}")
                return None

    except Exception as e:
        print(f"❌ [Generator] Exception during video generation: {e}")

    return None
