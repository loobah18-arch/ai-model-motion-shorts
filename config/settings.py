"""
Configuration settings for AI Model Motion Shorts Pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"

ASSETS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Development Mode (True = Generate & save locally, skip external uploads)
DEV_MODE = os.environ.get("DEV_MODE", "true").lower() in ("true", "1", "yes")

# API Keys (Optional, system defaults to 100% Free local engine)
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# Social Media Credentials (Used only when --upload flag is passed in production)
YOUTUBE_CLIENT_ID = os.environ.get("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET = os.environ.get("YOUTUBE_CLIENT_SECRET", "")
YOUTUBE_REFRESH_TOKEN = os.environ.get("YOUTUBE_REFRESH_TOKEN", "")

INSTAGRAM_USERNAME = os.environ.get("INSTAGRAM_USERNAME", "")
INSTAGRAM_PASSWORD = os.environ.get("INSTAGRAM_PASSWORD", "")
INSTAGRAM_ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_ACCOUNT_ID = os.environ.get("INSTAGRAM_ACCOUNT_ID", "")

# Default AI Motion Video Prompts for AI Model Videos
DEFAULT_MOTION_PROMPTS = [
    "cinematic slow motion turn, high fashion model, runway lighting, wind blowing hair, 4k ultra realistic",
    "stunning AI model posing for a luxury magazine shoot, subtle camera pan, natural lighting, ultra detailed skin texture",
    "aesthetic portrait shot, model looking gracefully at camera, elegant sunset glow, depth of field, slow motion 60fps",
    "cyberpunk futuristic fashion model posing under neon city lights, dynamic camera orbit, cinematic atmosphere",
    "minimalist editorial fashion model, sleek silhouette, subtle camera movement, 8k studio lighting"
]

# Video Output Specifications
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
VIDEO_DURATION_SEC = 5
