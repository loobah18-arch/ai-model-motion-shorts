# 🎬 AI Model Motion Shorts & Reels Automated Publisher

Automated pipeline to generate motion videos of your AI model using AI Video Generation APIs (Kling, Minimax, Luma, Replicate) and automatically schedule/publish them to **YouTube Shorts** and **Instagram Reels**.

---

## 🚀 Features

- **AI Image-to-Video Motion Generation**: Animates AI model photos into cinematic, high-fashion, or dynamic motion video clips using Replicate video generation APIs.
- **Auto 9:16 Vertical Crop & Processing**: Standardizes videos to 1080x1920 @ 30fps with FFmpeg, adding optional watermarks and audio mixing.
- **YouTube Shorts Auto-Publishing**: Directly uploads to YouTube Shorts via YouTube Data API v3.
- **Instagram Reels Auto-Publishing**: Supports both official Instagram Graph API and `instagrapi` session publisher.
- **24/7 Cloud Automation**: Daily cron scheduling via GitHub Actions (zero phone battery or machine upkeep).

---

## 📁 Repository Structure

```
ai-model-motion-shorts/
├── .github/workflows/
│   └── scheduled_publish.yml    # GitHub Actions daily schedule (0 12 * * *)
├── assets/                       # Put sample AI model photos here (.jpg / .png)
├── config/
│   └── settings.py               # Configuration & default prompt settings
├── core/
│   ├── generator.py              # Replicate AI video generation engine
│   └── editor.py                 # FFmpeg 9:16 vertical formatting engine
├── publishers/
│   ├── youtube_publisher.py      # YouTube Data API v3 Short uploader
│   └── instagram_publisher.py    # Instagram Reels Graph API & session uploader
├── main.py                       # Main pipeline CLI orchestrator
├── requirements.txt              # Python package requirements
└── .env.example                  # Environment variables template
```

---

## ⚡ Quick Start

### 1. Local Test Run

```bash
# Navigate to project directory
cd ~/antigravity/ai-model-motion-shorts

# Install requirements
pip install -r requirements.txt

# Run dry-run test (verifies structure without API calls)
python main.py --dry-run
```

### 2. Environment Setup

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required API Keys:
1. **Replicate API Token**: Get from [replicate.com](https://replicate.com/account/api-tokens)
2. **YouTube OAuth Credentials**: Create OAuth 2.0 Client ID in Google Cloud Console with YouTube Data API v3 enabled.
3. **Instagram Credentials**: Instagram handle & password or Graph API Access Token.

---

## ⏰ Automated Cloud Scheduling (GitHub Actions)

1. Push this repository to GitHub.
2. Go to **Settings > Secrets and variables > Actions** in your GitHub repository.
3. Add the following repository secrets:
   - `REPLICATE_API_TOKEN`
   - `YOUTUBE_CLIENT_ID`
   - `YOUTUBE_CLIENT_SECRET`
   - `YOUTUBE_REFRESH_TOKEN`
   - `INSTAGRAM_USERNAME`
   - `INSTAGRAM_PASSWORD`
4. The workflow in `.github/workflows/scheduled_publish.yml` will automatically run every day at 12:00 UTC and publish your AI model motion videos!
