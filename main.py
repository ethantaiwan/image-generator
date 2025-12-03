# main.py
import os
import re
import base64
import asyncio
import uuid
import json
import logging
import httpx
from typing import List, Optional, Dict, Any, Literal, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Query, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Google Gemini SDK
from google import genai
from google.genai import types

# --- 環境設定 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in env.")

# Gemini Client
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Gemini Client Init Error: {e}")

# Kling API (若有用到)
KLING_ACCESS_KEY = os.getenv("KLING_API_KEY")
KLING_SECRET_KEY = os.getenv("Secret_Key")

# 路徑設定
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENT_STORAGE_PATH = "/var/data"
os.makedirs(PERSISTENT_STORAGE_PATH, exist_ok=True)

IMAGE_PATHS = [f"00{i}.png" for i in range(1, 5)] # 預設 4 張
PUBLIC_URL_PREFIX = "/image-uploads/temp/"

# FastAPI
app = FastAPI(title="Video Gen Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ==========================================
#  Pydantic Models
# ==========================================
class ScriptPayload(BaseModel):
    result: str = Field(..., description="整段 storyboard 文字")
    images_per_prompt: int = Field(1, ge=1)
    start_index: int = Field(0, ge=0)
    naming: Literal["scene", "sequence"] = "scene"
    aspect_ratio: str = Field("16:9", description="圖片比例 e.g., 16:9, 9:16")

class FinalVideoConfig(BaseModel):
    prompt: Union[str, List[str]]
    image_urls: List[str]
    style: Literal["anime", "realistic"] = "realistic"
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    duration: Optional[float] = 5.0
    fps: Optional[int] = 24
    assemble: Optional[Literal["auto","copy_only"]] = "auto"
    transition_duration: Optional[float] = 0.5
    outro_duration: Optional[float] = 2.0
    model_name: Optional[str] = None
    mode: Optional[str] = None

class PromptRetrievalRequest(BaseModel):
    script: str

class PromptRetrievalResponse(BaseModel):
    video_prompts: List[str]

# Jobs 狀態存儲 (簡單版)
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

async def _set_job(job_id: str, **kv):
    async with JOBS_LOCK:
        job = JOBS.get(job_id, {})
        job.update(kv)
        job["updated_at"] = int(time.time())
        JOBS[job_id] = job

async def _get_job(job_id: str) -> Dict[str, Any] | None:
    async with JOBS_LOCK:
        return JOBS.get(job_id)

# ==========================================
#  輔助函數
# ==========================================
def parse_image_prompts(text: str) -> List[str]:
    text = text.replace('\r\n', '\n')
    marker = re.compile(r'(?i)(image[\s\_]*prompt.*?)[:：]\s*', flags=re.DOTALL)
    stop_line = re.compile(r'^\s*(?:Scene\s*\d+|[0-9０-９]+\)|\[.*?\])', flags=re.IGNORECASE)
    prompts = []
    for m in marker.finditer(text):
        start = m.end()
        next_m = marker.search(text, pos=start)
        chunk = text[start: next_m.start()] if next_m else text[start:]
        lines = chunk.split('\n')
        buf = []
        for line in lines:
            if not line.strip() or stop_line.match(line): break
            buf.append(line.strip())
        if buf: prompts.append(' '.join(buf).strip())
    return prompts

def extract_video_prompts(script_text: str) -> List[str]:
    if not script_text: return []
    prompts = []
    seen = set()
    # 寬鬆版 Regex
    pattern = r"(?:【|\b)video[\_\s-]*prompt.*?[:：】]\s*(.+?)(?=\n\s*(?:Scene|image_|【|\d+[.)])|\Z)"
    matches = list(re.finditer(pattern, script_text, flags=re.DOTALL | re.IGNORECASE))
    for m in matches:
        vp = m.group(1).strip()
        if vp and vp not in seen:
            seen.add(vp)
            prompts.append(vp)
    return prompts

async def save_image_to_disk(img_data: str, index: int) -> Union[str, None]:
    filename = IMAGE_PATHS[index] if index < len(IMAGE_PATHS) else f"extra_{index}.png"
    full_path = os.path.join(PERSISTENT_STORAGE_PATH, filename)
    try:
        if img_data.startswith("data:image/"):
            base64_content = img_data.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_content)
        else:
            image_bytes = base64.b64decode(img_data)
        
        await asyncio.to_thread(lambda: open(full_path, "wb").write(image_bytes))
        return PUBLIC_URL_PREFIX + filename
    except Exception as e:
        print(f"[Save Error] {e}")
        return None

# -------------------------------------------------------
#  Gemini 生圖核心 (含尺寸強制)
# -------------------------------------------------------
def gemini_image_generation(prompt: str, count: int = 1, aspect_ratio: str = "16:9") -> List[str]:
    """
    使用 gemini-2.5-flash-image 進行文生圖。
    修正：使用 image_config 正確傳遞比例參數。
    """
    # 確保有模型名稱
    model = os.getenv("model_name", "gemini-2.5-flash-image") 
    
    print(f"[DEBUG] Model: {model} | Ratio: {aspect_ratio}")
    
    urls: List[str] = []
    
    # 依需求產生多張
    for _ in range(max(1, count)):
        try:
            # 呼叫 API
            resp = client.models.generate_content(
                model=model,
                contents=[prompt], # Prompt 不用再加那一長串尺寸指令了，沒用
                config=types.GenerateContentConfig(
                    response_modalities=["Image"],        
                    temperature=0.8,
                    
                    # ★★★ 關鍵修正：把比例放在 image_config 裡面 ★★★
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio  # 支援 "16:9", "9:16", "4:3", "3:4", "1:1"
                    ),
                ),
            )

            # 解析回應 (保持原本邏輯)
            parts = getattr(resp.candidates[0].content, "parts", []) if resp.candidates else []
            for p in parts:
                inline = getattr(p, "inline_data", None)
                if inline and getattr(inline, "mime_type", "").startswith("image/"):
                    data = inline.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    b64 = base64.b64encode(data).decode("utf-8")
                    mime = inline.mime_type or "image/png"
                    urls.append(f"data:{mime};base64,{b64}")
                    
        except Exception as e:
            # 印出詳細錯誤以便除錯
            print(f"[Gemini Error] {e}")
            # 如果是因為參數錯誤 (例如 4:3 不支援)，可以嘗試 fallback 到 16:9
            if "aspect_ratio" in str(e) and aspect_ratio != "16:9":
                print("Retrying with default 16:9...")
                return gemini_image_generation(prompt, count, "16:9")
            continue

    # 去重
    dedup, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)

    return dedup[:count]


# ==========================================
#  API 路由
# ==========================================
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.get(PUBLIC_URL_PREFIX + "{filename}")
async def serve_image(filename: str):
    full_path = os.path.join(PERSISTENT_STORAGE_PATH, filename)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    raise HTTPException(404, "Image not found")

@app.post("/prompt-retrieval", response_model=PromptRetrievalResponse)
def prompt_retrieval(req: PromptRetrievalRequest):
    prompts = extract_video_prompts(req.script)
    if not prompts:
        raise HTTPException(422, "找不到 video_prompt")
    return PromptRetrievalResponse(video_prompts=prompts)

@app.post("/extract_then_generate")
async def extract_then_generate(payload: ScriptPayload):
    text = payload.result.strip()
    prompts = parse_image_prompts(text)
    if not prompts:
        raise HTTPException(422, "找不到 image_prompt")

    results = []
    current_index = payload.start_index
    
    # 傳遞 aspect_ratio
    ratio = payload.aspect_ratio 

    for i, p in enumerate(prompts):
        try:
            images = gemini_image_generation(p, count=1, aspect_ratio=ratio)
            if not images:
                raise ValueError("無圖片返回")
            
            url = await save_image_to_disk(images[0], current_index)
            results.append({
                "prompt": p, 
                "uploaded_urls": [url], 
                "errors": []
            })
            current_index += 1
        except Exception as e:
            results.append({
                "prompt": p, 
                "uploaded_urls": [], 
                "errors": [str(e)]
            })

    uploaded_flat = [r["uploaded_urls"][0] for r in results if r["uploaded_urls"]]
    
    return {
        "generate_result": {"results": results},
        "uploaded_urls_flat": uploaded_flat,
        "n_prompts": len(prompts)
    }

# -------------------------------------------------------
#  影片生成 Pipeline (非同步 + 修正版)
# -------------------------------------------------------
def _clean_prompt_backend(text: str) -> str:
    if not text: return ""
    noise = [r"storyboard_text", r"純文字腳本", r"Note:", r"注意：", r"End of script"]
    cleaned = text
    for pat in noise:
        match = re.search(pat, cleaned, flags=re.IGNORECASE)
        if match: cleaned = cleaned[:match.start()]
    return cleaned.strip()

async def _run_final_video_pipeline(job_id: str, cfg: FinalVideoConfig) -> dict:
    # 這裡請填入你的 Kling API 呼叫邏輯、FFmpeg 合併邏輯
    # 為了節省篇幅，我只列出修正後的關鍵結構 (請把你的 _run_final_video_pipeline 內容填回來，並套用以下修正)
    
    print(f"[JOB {job_id}] Pipeline Start. Size: {cfg.width}x{cfg.height}")
    
    # 1. 清洗 Prompts
    clean_prompts = []
    raw_prompts = cfg.prompt if isinstance(cfg.prompt, list) else [cfg.prompt] * len(cfg.image_urls)
    
    for p in raw_prompts:
        clean_prompts.append(_clean_prompt_backend(p))
        
    # 2. 執行 Kling 任務 (略，請照你原本的寫，記得 width=cfg.width)
    
    # 3. FFmpeg 合併 (修正點：不要寫死 1920x1080)
    # _concat_with_xfade(..., width=cfg.width, height=cfg.height)
    
    return {"final_video_url": f"/outputs/{job_id}/final.mp4"}

async def _run_wrapper(job_id: str, cfg: FinalVideoConfig):
    try:
        await _set_job(job_id, status="running")
        res = await _run_final_video_pipeline(job_id, cfg)
        await _set_job(job_id, status="succeed", result=res)
    except Exception as e:
        print(f"[Job Failed] {e}")
        await _set_job(job_id, status="failed", error=str(e))

@app.post("/generate-final-video")
async def generate_final_video(cfg: FinalVideoConfig, bg_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    await _set_job(job_id, status="queued")
    bg_tasks.add_task(_run_wrapper, job_id, cfg)
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await _get_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return job

import time
