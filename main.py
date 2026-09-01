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
from core.pose_motion_transfer import transfer_motion_to_model
from core.editor import format_video_for_vertical_shorts
from publishers.youtube_publisher import upload_to_youtube
from publishers.instagram_publisher import upload_to_instagram

def run_pipeline(
    input_image: Optional[Path] = None,
    motion_ref: Optional[Path] = None,
    prompt: Optional[str] = None,
    watermark: str = "@AIModelOfficial",
    transfer_motion: bool = False,
    batch: bool = False,
    allow_upload: bool = False
):
    print("==================================================")
    print("🤖 AI MODEL MOTION PIPELINE (DEV PHASE)")
    print("==================================================")

    # 1. Select target AI model photo
    if not input_image or not input_image.exists():
        primary_model = ASSETS_DIR / "model_primary.jpg"
        if primary_model.exists():
            input_image = primary_model
            print(f"📸 Using primary AI model photo: {input_image.name}")
        else:
            images = list(ASSETS_DIR.glob("*.jpg")) + list(ASSETS_DIR.glob("*.png"))
            if images:
                input_image = random.choice(images)
                print(f"📸 Using AI model photo from assets: {input_image.name}")
            else:
                print("❌ No model photo found in assets/! Please place your AI model photo in assets/model_primary.jpg.")
                sys.exit(1)

    if not prompt:
        prompt = random.choice(DEFAULT_MOTION_PROMPTS)

    title = "AI Model Motion #Shorts #Reels"
    caption = f"{prompt}\n\n#AIModel #AIVideo #Shorts #Reels #ModelFashion #DigitalCreator"
    tags = ["AI Model", "AI Video", "Shorts", "Reels", "Fashion Model", "Digital Human"]

    # 2. Pose & Dance Motion Transfer Mode
    if transfer_motion:
        print(f"💃 [Pose Motion Transfer Mode] Animating AI Model {input_image.name} with reference dance video...")
        raw_video = transfer_motion_to_model(model_image_path=input_image, reference_video_path=motion_ref)
        if not raw_video:
            raw_video = create_free_ffmpeg_motion(image_path=input_image, motion_type="zoom_in")
            
        final_video = format_video_for_vertical_shorts(input_video=raw_video, watermark_text=watermark)
        print("\n==================================================")
        print("🎉 POSE & DANCE MOTION TRANSFER COMPLETE")
        print(f"📁 Output Video: {final_video}")
        print("==================================================")
        return

    # 3. Batch Generation vs Single Generation
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

    # 4. Format & Edit Video to 9:16 Vertical Ratio
    final_video = format_video_for_vertical_shorts(input_video=raw_video, watermark_text=watermark)
    if not final_video:
        final_video = raw_video

    # 5. Upload Control
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
    ref_path = Path(args.motion_ref) if args.motion_ref else None
    run_pipeline(
        input_image=img_path,
        motion_ref=ref_path,
        prompt=args.prompt,
        watermark=args.watermark,
        transfer_motion=args.transfer_motion,
        batch=args.batch,
        allow_upload=args.upload
    )
