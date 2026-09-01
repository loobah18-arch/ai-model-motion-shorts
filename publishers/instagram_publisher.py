"""
Instagram Reels Auto-Publisher.
Supports Instagram Graph API (Official Business/Creator API) & instagrapi session login fallback.
"""
import os
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional

def upload_to_instagram_graph_api(
    video_url: str,
    caption: str,
    access_token: str,
    ig_account_id: str
) -> Dict[str, Any]:
    """
    Publishes a Reel via official Instagram Graph API.
    Note: Requires a publicly accessible video_url.
    """
    print(f"🚀 [Instagram Graph API] Container creation for Reel...")
    
    # 1. Create Media Container
    container_url = f"https://graph.facebook.com/v19.0/{ig_account_id}/media"
    container_payload = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": access_token
    }
    
    try:
        res = requests.post(container_url, data=container_payload, timeout=30)
        if res.status_code != 200:
            print(f"❌ [Instagram Graph API] Container error: {res.text}")
            return {"status": "error", "error": res.text}
            
        container_id = res.json().get("id")
        print(f"⏳ [Instagram Graph API] Media container created ({container_id}). Waiting for encoding...")
        
        # 2. Check Container Status until READY
        status_url = f"https://graph.facebook.com/v19.0/{container_id}"
        for _ in range(30):
            time.sleep(5)
            check_res = requests.get(status_url, params={"fields": "status_code", "access_token": access_token})
            status_code = check_res.json().get("status_code")
            print(f"   Reel Status: {status_code}")
            
            if status_code == "FINISHED":
                # 3. Publish Media Container
                publish_url = f"https://graph.facebook.com/v19.0/{ig_account_id}/media_publish"
                pub_res = requests.post(publish_url, data={"creation_id": container_id, "access_token": access_token}, timeout=30)
                if pub_res.status_code == 200:
                    media_id = pub_res.json().get("id")
                    print(f"✅ [Instagram Graph API] Reel published successfully! Media ID: {media_id}")
                    return {"status": "success", "media_id": media_id}
                else:
                    return {"status": "error", "error": pub_res.text}
            elif status_code == "ERROR":
                return {"status": "error", "error": "Reel processing failed on Instagram servers"}
                
    except Exception as e:
        print(f"❌ [Instagram Graph API] Exception: {e}")
        return {"status": "error", "error": str(e)}

    return {"status": "error", "error": "Timeout waiting for video encoding"}


def upload_to_instagram_instagrapi(
    video_path: Path,
    caption: str,
    username: str,
    password: str
) -> Dict[str, Any]:
    """
    Fallback publisher using instagrapi for accounts without Graph API setup.
    """
    try:
        from instagrapi import Client
        cl = Client()
        
        session_file = Path("config/instagram_session.json")
        if session_file.exists():
            print("🔑 [Instagram] Loading saved session...")
            cl.load_settings(session_file)
            
        print(f"🔑 [Instagram] Logging in as {username}...")
        cl.login(username, password)
        cl.dump_settings(session_file)
        
        print(f"🚀 [Instagram] Uploading Reel: {video_path.name}...")
        media = cl.clip_upload(str(video_path), caption=caption)
        
        reel_url = f"https://www.instagram.com/reel/{media.code}/"
        print(f"✅ [Instagram] Reel uploaded successfully! URL: {reel_url}")
        return {
            "status": "success",
            "media_id": media.pk,
            "url": reel_url
        }
    except Exception as e:
        print(f"❌ [Instagram instagrapi] Upload failed: {e}")
        return {"status": "error", "error": str(e)}


def upload_to_instagram(
    video_path: Path,
    caption: str,
    public_video_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main Instagram uploader wrapper. Prefers Graph API if configured,
    falls back to instagrapi.
    """
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    ig_account_id = os.environ.get("INSTAGRAM_ACCOUNT_ID")
    
    if access_token and ig_account_id and public_video_url:
        return upload_to_instagram_graph_api(public_video_url, caption, access_token, ig_account_id)
        
    username = os.environ.get("INSTAGRAM_USERNAME")
    password = os.environ.get("INSTAGRAM_PASSWORD")
    
    if username and password:
        return upload_to_instagram_instagrapi(Path(video_path), caption, username, password)
        
    print("⚠️ [Instagram] No credentials found (INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_USERNAME/PASSWORD). Skipping.")
    return {"status": "skipped", "reason": "No credentials"}
