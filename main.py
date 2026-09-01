"""
Main Orchestration Script for AI Model Motion Shorts Pipeline.
Development Phase Edition: Generates and previews videos locally ($0 Cost) without uploading.
"""
import argparse
import sys
import os
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from config.settings import DEFAULT_MOTION_PROMPTS, ASSETS_DIR, DEV_MODE, REPLICATE_API_TOKEN
from core.generator import generate_motion_video
from core.free_motion import create_free_ffmpeg_motion, batch_generate_dev_motions, CATALOG_MD
from core.editor import format_video_for_vertical_shorts
from publishers.youtube_publisher import upload_to_youtube
from publishers.instagram_publisher import upload_to_instagram

def run_pipeline(
    input_image: Optional[Path] = None,
    prompt: Optional[str] = None,
    watermark: str = "@AIModelOfficial",
    batch: bool = False,
    allow_upload: bool = False
):
    print("==================================================")
    print("🤖 AI MODEL MOTION PIPELINE (DEV PHASE)")
    print("==================================================")

    # 1. Look for AI model photo in assets directory or generate demo image
    if not input_image or not input_image.exists():
        images = list(ASSETS_DIR.glob("*.jpg")) + list(ASSETS_DIR.glob("*.png")) + list(ASSETS_DIR.glob("*.jpeg"))
        if images:
            input_image = random.choice(images)
            print(f"📸 Using AI model photo from assets: {input_image.name}")
        else:
            # Create sample demonstration image if none exists
            sample_img = ASSETS_DIR / "sample_model.jpg"
            if not sample_img.exists():
                print("🎨 Creating sample model image asset for testing...")
                from PIL import Image, ImageDraw
                img = Image.new("RGB", (1080, 1920), color=(20, 24, 40))
                draw = ImageDraw.Draw(img)
                draw.rectangle([200, 400, 880, 1500], fill=(60, 90, 180))
                draw.text((380, 900), "AI MODEL PHOTO", fill=(255, 255, 255))
                img.save(sample_img)
            input_image = sample_img

    if not prompt:
        prompt = random.choice(DEFAULT_MOTION_PROMPTS)

    title = "AI Model Motion #Shorts #Reels"
    caption = f"{prompt}\n\n#AIModel #AIVideo #Shorts #Reels #ModelFashion #DigitalCreator"
    tags = ["AI Model", "AI Video", "Shorts", "Reels", "Fashion Model", "Digital Human"]

    # 2. Batch Generation vs Single Generation
    if batch:
        generated_files = batch_generate_dev_motions(input_image)
        print("\n==================================================")
        print(f"🎉 BATCH DEV GENERATION COMPLETE: {len(generated_files)} videos generated!")
        print(f"📋 Local Video Catalog: {CATALOG_MD}")
        print("==================================================")
        return

    # Single Motion Generation
    raw_video = None
    if REPLICATE_API_TOKEN:
        print("⚡ Paid Replicate API Token detected. Using cloud GPU model...")
        raw_video = generate_motion_video(image_path=input_image, prompt=prompt)

    if not raw_video:
        print("💡 [100 percent FREE ENGINE] Generating local phone motion clip ($0 cost)...")
        raw_video = create_free_ffmpeg_motion(
            image_path=input_image,
            motion_type=random.choice(["zoom_in", "pan_down", "pan_up", "subtle_orbit", "parallax_shift"])
        )

    if not raw_video:
        print("❌ Pipeline failed during video motion step.")
        sys.exit(1)

    # 3. Format & Edit Video to 9:16 Vertical Ratio
    final_video = format_video_for_vertical_shorts(input_video=raw_video, watermark_text=watermark)
    if not final_video:
        final_video = raw_video

    # 4. Upload Control (Skipped during Dev Phase unless --upload passed)
    if not allow_upload:
        print("\n🔒 [DEV PHASE] Uploads are currently DISABLED.")
        print(f"📁 Local Video Ready for Preview: {final_video}")
        print(f"📝 Local Catalog Updated: {CATALOG_MD}")
        print("   (To enable live publishing later, pass --upload flag)")
        return

    # Uploads enabled
    yt_res = upload_to_youtube(video_path=final_video, title=title, description=caption, tags=tags)
    ig_res = upload_to_instagram(video_path=final_video, caption=caption)

    print("\n==================================================")
    print("🎉 PIPELINE EXECUTION COMPLETED")
    print(f"Video File:       {final_video.name}")
    print(f"YouTube Result:   {yt_res.get('status')}")
    print(f"Instagram Result: {ig_res.get('status')}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Motion Shorts Pipeline (Dev Phase Edition)")
    parser.add_argument("--image", type=str, help="Path to input AI model photo")
    parser.add_argument("--motion-ref", type=str, help="Path to reference dance/movement video")
    parser.add_argument("--prompt", type=str, help="Motion generation prompt")
    parser.add_argument("--watermark", type=str, default="@AIModelOfficial", help="Watermark text")
    parser.add_argument("--transfer-motion", action="store_true", help="Perform AI Pose & Dance Motion Transfer from reference video")
    parser.add_argument("--batch", action="store_true", help="Generate all motion presets (zoom, pan, orbit, parallax)")
    parser.add_argument("--upload", action="store_true", help="Enable live social media publishing (disabled in dev phase by default)")

    args = parser.parse_args()
    img_path = Path(args.image) if args.image else None
    run_pipeline(
        input_image=img_path,
        prompt=args.prompt,
        watermark=args.watermark,
        batch=args.batch,
        allow_upload=args.upload
    )
