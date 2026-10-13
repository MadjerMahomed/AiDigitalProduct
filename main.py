import os
import json
import uuid
import traceback
import textwrap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
from duckduckgo_search import DDGS
from PIL import Image, ImageDraw
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(OS_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Persistent database file for product history
DB_FILE = os.path.join(OS_DIR, "products_db.json")

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

active_product_id = None

class AgentRequest(BaseModel):
    topic: str = ""

class SwitchRequest(BaseModel):
    product_id: str

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(OS_DIR, "templates", "index.html"))

@app.get("/api/products")
def get_products():
    return {"status": "success", "products": load_db(), "active_id": active_product_id}

@app.post("/api/switch-product")
def switch_product(req: SwitchRequest):
    global active_product_id
    db = load_db()
    prod = next((p for p in db if p["id"] == req.product_id), None)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")
    active_product_id = prod["id"]
    return {"status": "success", "data": prod}

@app.post("/api/run-agent")
def run_agent(req: AgentRequest):
    global active_product_id
    try:
        if req.topic.strip():
            snippets = [f"User requested niche/topic: {req.topic.strip()}"]
        else:
            try:
                with DDGS() as ddg:
                    search_results = list(ddg.text("trending digital product templates code spreadsheets guide", max_results=3))
                snippets = [res.get('body', '') for res in search_results]
            except Exception as search_err:
                print("Search warning (using default trends):", search_err)
                snippets = ["Notion productivity templates", "Python automation scripts", "UI design kits"]

        prompt = f"""
        Analyze these inputs/topics: {snippets}
        Select ONE high-value digital product idea that can be delivered as a comprehensive guide, planner template, or script.

        You MUST respond strictly in valid JSON format with the following keys:
        - "product_title": Short title of the product
        - "product_description": Detailed summary/content of the digital product
        - "full_content": Complete, ready-to-use text content / layout / planner framework for the actual digital product file (write detailed sections, instructions, or code templates).
        - "video_hook": A 5-7 word eye-catching text hook for a video reel/TikTok
        - "caption": Engaging Instagram/TikTok caption with emojis and relevant hashtags
        """

        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content'].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {
                "product_title": req.topic.strip() if req.topic.strip() else "Ultimate Notion Life Planner",
                "product_description": content,
                "full_content": content,
                "video_hook": "Stop wasting time, use this template!",
                "caption": "Level up your productivity with this template! 🚀 #productivity"
            }

        prod_id = str(uuid.uuid4())[:8]
        assets = generate_media_assets(prod_id, data)

        product_record = {
            "id": prod_id,
            "product_title": data.get("product_title"),
            "product_description": data.get("product_description"),
            "full_content": data.get("full_content", data.get("product_description")),
            "video_hook": data.get("video_hook"),
            "caption": data.get("caption"),
            "image_url": assets["image_url"],
            "video_url": assets["video_url"],
            "file_url": assets["file_url"]
        }

        db = load_db()
        db.insert(0, product_record) # Add to top of list
        save_db(db)
        active_product_id = prod_id

        return {"status": "success", "data": product_record}

    except Exception as e:
        print("--- ERROR IN AGENT PIPELINE ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-variations")
def generate_variations():
    global active_product_id
    db = load_db()
    prod = next((p for p in db if p["id"] == active_product_id), None)
    if not prod:
        raise HTTPException(status_code=400, detail="No active product selected.")
    
    try:
        prompt = f"""
        We have this digital product:
        Title: {prod['product_title']}
        Description: {prod['product_description']}

        Generate a NEW alternative short video hook (5-7 words) and a NEW alternative social media caption.
        You MUST respond strictly in valid JSON format with keys:
        - "video_hook": Alternative catchy text hook for TikTok/Reels
        - "caption": Alternative engaging Instagram/TikTok caption with emojis and hashtags
        """

        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content'].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        try:
            vars_data = json.loads(content)
            prod["video_hook"] = vars_data.get("video_hook", prod["video_hook"])
            prod["caption"] = vars_data.get("caption", prod["caption"])
        except json.JSONDecodeError:
            pass

        # Re-render assets with new hook
        assets = generate_media_assets(prod["id"], prod)
        prod["video_url"] = assets["video_url"]

        save_db(db)
        return {"status": "success", "data": prod}

    except Exception as e:
        print("--- ERROR IN VARIATIONS ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def generate_media_assets(prod_id, data):
    # Unique suffix to avoid overwriting files between products
    safe_prefix = f"prod_{prod_id}"

    # 1. Save actual digital product file (.md / planner document)
    file_filename = f"{safe_prefix}_planner.md"
    file_path = os.path.join(OUTPUT_DIR, file_filename)
    file_text = f"# {data.get('product_title')}\n\n## Overview\n{data.get('product_description')}\n\n## Complete Digital Content & Framework\n{data.get('full_content', '')}"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_text)

    # 2. Generate Cover Image
    cover_img = Image.new('RGB', (1080, 1080), color=(15, 23, 42))
    draw_cover = ImageDraw.Draw(cover_img)
    draw_cover.rectangle([60, 60, 1020, 1020], outline=(30, 41, 59), width=4)
    draw_cover.rectangle([100, 100, 980, 300], fill=(30, 41, 59))
    draw_cover.text((130, 130), "🔥 DIGITAL PLANNER & ASSET", fill=(56, 189, 248))
    
    title_lines = textwrap.wrap(data.get('product_title', 'Digital Product'), width=25)
    y_text = 400
    for line in title_lines:
        draw_cover.text((100, y_text), line, fill=(255, 255, 255))
        y_text += 70

    draw_cover.text((100, 900), "READY-TO-USE TEMPLATE FILE", fill=(148, 163, 184))
    image_filename = f"{safe_prefix}_cover.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    cover_img.save(image_path)

    # 3. Generate Voiceover
    voice_path = os.path.join(OUTPUT_DIR, f"{safe_prefix}_voice.mp3")
    tts = gTTS(text=data.get('video_hook', 'Check this out'), lang='en')
    tts.save(voice_path)

    # 4. Render Video Frame
    frame_img = Image.new('RGB', (1080, 1920), color=(11, 15, 25))
    draw_frame = ImageDraw.Draw(frame_img)
    draw_frame.rectangle([80, 150, 1000, 280], fill=(30, 41, 59), outline=(56, 189, 248), width=2)
    draw_frame.text((130, 185), "⚡ EXCLUSIVE PLANNER DROP", fill=(56, 189, 248))

    draw_frame.rectangle([80, 350, 1000, 850], fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    draw_frame.text((120, 400), "PRODUCT BLUEPRINT", fill=(148, 163, 184))
    
    vid_title_lines = textwrap.wrap(data.get('product_title', 'Product'), width=22)
    y_vid_title = 500
    for line in vid_title_lines:
        draw_frame.text((120, y_vid_title), line, fill=(255, 255, 255))
        y_vid_title += 80

    draw_frame.rectangle([80, 1000, 1000, 1450], fill=(30, 27, 75), outline=(168, 85, 247), width=3)
    draw_frame.text((120, 1050), "💬 VIDEO HOOK", fill=(216, 180, 254))
    
    hook_lines = textwrap.wrap(f'"{data.get("video_hook", "Check this out")}"', width=20)
    y_hook = 1150
    for line in hook_lines:
        draw_frame.text((120, y_hook), line, fill=(255, 255, 255))
        y_hook += 75

    draw_frame.text((360, 1780), "Created with Local AI Agent", fill=(100, 116, 139))
    video_frame_path = os.path.join(OUTPUT_DIR, f"{safe_prefix}_frame.png")
    frame_img.save(video_frame_path)

    # 5. Assemble Video
    audio_clip = AudioFileClip(voice_path)
    duration = max(4, audio_clip.duration + 1)
    video = ImageClip(video_frame_path).set_duration(duration).set_audio(audio_clip)
    video_filename = f"{safe_prefix}_video.mp4"
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    video.write_videofile(video_path, fps=24, codec='libx264', audio_codec='aac')

    return {
        "image_url": f"/output/{image_filename}",
        "video_url": f"/output/{video_filename}",
        "file_url": f"/output/{file_filename}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)