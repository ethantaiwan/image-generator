# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import json, os
import base64
import uuid

# --- 環境變數設定和初始化 ---
# 確保 GOOGLE_API_KEY 是您的環境變數名稱
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
import os
import json
import uuid
import base64
from typing import Any, Dict, List, Union, Optional
import nest_asyncio
import re
import io
# --- FastAPI 和 Pydantic 相關匯入 ---
from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Gemini API 相關匯入 ---
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Uvicorn 和 Asyncio 相關匯入 (用於 Notebook 啟動) ---

import httpx 


# ==========================================================
# ⚡️ 核心設定區塊
# ==========================================================

# 確保 API Key 存在
# ⚠️ 請將 'YOUR_GOOGLE_API_KEY' 替換為您環境變數的名稱，或直接設置
try:
    if not GOOGLE_API_KEY:
         # 如果環境變數未設定，您可以手動在這裡填入您的 KEY 進行測試
         # ⚠️ 僅用於測試，生產環境請使用環境變數
         # GOOGLE_API_KEY = "AIzaSy..."
         if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY 環境變數未設定。")
except Exception as e:
     # 如果您在 Colab/Jupyter 中運行，可能需要手動定義 GOOGLE_API_KEY 
     # 否則這行程式碼會因為找不到變數而報錯
     # 假設您在 Colab/Jupyter 中已經定義了 GOOGLE_API_KEY
     print("API Key 配置跳過環境變數檢查，請確保變數 GOOGLE_API_KEY 已存在於您的執行環境中。")
     # 為了讓程式碼通過，這裡假設 GOOGLE_API_KEY 變數已經在 Notebook 前面定義了。


# Gemini 初始化
client = genai.Client(api_key=GOOGLE_API_KEY)

# 使用者指定的模型
MODEL_NAME = os.getenv("model_name") 


#try:
#    response = client.models.generate_content(
#        model="gemini-2.5-flash-preview-09-2025",
#        contents=["說你好"],
#    )
#    print("API 連線成功，文字輸出:", response.text)
#except Exception as e:
#    print("API 連線失敗:", e)
# --- FastAPI 應用初始化 ---
app = FastAPI()

# --- CORS 中間件配置 (解決前端 'Failed to fetch' 問題) ---
origins = ["*"] # 允許所有來源 (用於測試)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,      
    allow_methods=["*"],         
    allow_headers=["*"],         
)
# ==========================================================
# ⚙️ 數據模型與輔助函數
# ==========================================================

# 數據模型 (Pydantic)
class KontextAndImageCreate(BaseModel):
    user_id: str
    character_name: str
    description: str
    base_prompt: Optional[str] = None
    image_count: int = 1 # 由於 generate_content 限制，這裡預設改為 1

class ImageBatchResponse(BaseModel):
    full_prompt: str
    image_urls: List[str]

# 輔助函數 (為符合您的要求，此函數使用 client.models.generate_content)
def gemini_image_generation(prompt: str, count: int = 1) -> List[str]:
    """
    使用 gemini-2.5-flash-image 進行文生圖，回傳 Base64 Data URL。
    注意：一次呼叫通常只會回一張，若要多張就 loop。
    """
    #model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
    model = os.getenv("model_name") 

    urls: List[str] = []

    # 依需求產生多張
    for _ in range(max(1, count)):
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            # 關鍵：指定只回 Image，避免文字吞掉輸出；需要新版本 google-genai
            config=types.GenerateContentConfig(
                response_modalities=["Image"],        # ← 只回圖片
                # 可選：設定比例（官方文件支援 image_config.aspect_ratio）
                # image_config=types.ImageConfig(aspect_ratio="1:1"),
                temperature=0.8,
            ),
        )

        # 正確解析路徑：candidates[0].content.parts
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

    # 去重＋裁切
    dedup, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup[:count]

from google.genai import types
import base64
from typing import List, Optional

# 假設 client 和 MODEL_NAME="gemini-2.5-flash-image-preview" 已經定義

def gemini_image_editing(
    edit_prompt: str,
    original_image_bytes: bytes,
    image_mime_type: str = "image/jpeg"
) -> Optional[str]:
    #model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
    model = os.getenv("model_name") 

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=original_image_bytes, mime_type=image_mime_type),
            {"text": edit_prompt},
        ],
        config=types.GenerateContentConfig(
            response_modalities=["Image"],
            # 可選：image_config=types.ImageConfig(aspect_ratio="1:1"),
        ),
    )

    parts = getattr(resp.candidates[0].content, "parts", []) if resp.candidates else []
    for p in parts:
        inline = getattr(p, "inline_data", None)
        if inline and getattr(inline, "mime_type", "").startswith("image/"):
            data = inline.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            b64 = base64.b64encode(data).decode("utf-8")
            mime = inline.mime_type or "image/png"
            return f"data:{mime};base64,{b64}"
    return None

# ==========================================================
# 🚀 API 路由定義
# ==========================================================

@app.get("/")
def read_root():
    return {"status": "ok", "message": f"FastAPI Server is running. Model: {MODEL_NAME}"}

@app.post("/create_kontext_and_generate", response_model=ImageBatchResponse)
def create_kontext_and_generate(payload: KontextAndImageCreate):
    
    # 組合提示詞
    base_prompt = payload.base_prompt if payload.base_prompt else ""
    full_prompt = f"{payload.description}. {base_prompt}"
    
    # 獲取 Base64 Data URLs
    images = gemini_image_generation(full_prompt, count=payload.image_count)

    if not images:
        # 如果 gemini_image_generation 返回空列表
        raise HTTPException(
            status_code=500, 
            detail="Gemini generation failed or no image data returned. Please check the model's capability and API Key."
        )
        
    # 由於我們移除了文件持久化，這裡只返回生成的圖像
    return ImageBatchResponse(full_prompt=full_prompt, image_urls=images)

@app.post("/edit_image")
async def edit_image_api(
    edit_prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """
    呼叫 gemini_image_editing 進行圖片修改。
    前端上傳圖片與提示詞即可，例如：
    FormData:
      - edit_prompt: "讓畫面更明亮，保持手繪質感"
      - file: <image>
    """

    try:
        # 讀取上傳的圖片 bytes
        original_image_bytes = await file.read()
        image_mime_type = file.content_type or "image/jpeg"

        # 呼叫你原本的函式
        edited_image_data_url = gemini_image_editing(
            edit_prompt=edit_prompt,
            original_image_bytes=original_image_bytes,
            image_mime_type=image_mime_type
        )

        if not edited_image_data_url:
            raise HTTPException(
                status_code=500,
                detail="Gemini 沒有返回圖片資料，請檢查模型權限或提示詞。"
            )

        return {
            "edit_prompt": edit_prompt,
            "image_url": edited_image_data_url
        }


# ❗ 修正：新增 except 區塊來處理錯誤 ❗
    except HTTPException as e:
        # 捕捉您自己拋出的 HTTP 錯誤
        raise e
    except Exception as e:
        # 捕捉其他所有未預期的錯誤，例如檔案讀取失敗、API 連線錯誤等
        raise HTTPException(
            status_code=500,
            detail=f"圖片編輯處理失敗: {str(e)}"
        )

# --- Render Persistent Disk 設定 ---
# 這是您在 Render 儀表板設定的掛載點
PERSISTENT_STORAGE_PATH = "/var/data" 
MAX_IMAGES = 4
IMAGE_PATHS = [f"00{i}.png" for i in range(1, MAX_IMAGES + 1)]
PUBLIC_URL_PREFIX = "/image-uploads/temp/"

# --- 假設遠端服務的 URL ---
# 請將這裡替換成您實際部署 image-generator 的 API 地址
REMOTE_IMAGE_GENERATOR_URL = "https://image-generator-i03j.onrender.com/api/image-generate" 


# --- 輔助函式：JSON 圖片字串提取 (根據您的要求) ---

def looks_like_img_url(s: str) -> bool:
    """粗略判斷字串是否為圖片連結或 Base64 字串"""
    s = s.strip()
    return (
        s.startswith("data:image/") or
        s.startswith("http://") or s.startswith("https://") or
        (re.fullmatch(r"[A-Za-z0-9+/=\s]+", s or "") and len(s) > 100)
    )

def find_image_strings(obj: Union[Dict, List]) -> List[str]:
    """遞迴地在複雜的 JSON 結構中尋找圖片連結或 Base64 字串"""
    found = []
    if isinstance(obj, dict):
        for k in ["image_url", "image", "url", "image_urls", "images", "urls", "results"]:
            if k in obj:
                value = obj[k]
                if isinstance(value, str) and looks_like_img_url(value):
                    found.append(value)
                elif isinstance(value, (list, dict)):
                    found.extend(find_image_strings(value))
        # 遞迴其他鍵
        for v in obj.values():
            if isinstance(v, (list, dict)):
                found.extend(find_image_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str) and looks_like_img_url(v):
                 found.append(v)
            elif isinstance(v, (list, dict)):
                found.extend(find_image_strings(v))
    return found


# --- Pydantic 模型用於請求 Body ---
class GeneratorRequest(BaseModel):
    """用於接收遠端服務返回的 JSON 結構"""
    # 這裡假設遠端服務回傳一個 JSON，結構不固定，但會包含圖片連結
    data: Any


# --- 圖片儲存和處理邏輯 ---

async def fetch_and_save_image(img_data: str, index: int) -> Union[str, None]:
    """將 Base64 或 URL 圖片下載並儲存到持久性磁碟"""
    filename = IMAGE_PATHS[index]
    full_path = os.path.join(PERSISTENT_STORAGE_PATH, filename)
    
    try:
        if img_data.startswith("data:image/"):
            # 處理 Base64
            base64_content = img_data.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_content)
        elif img_data.startswith(("http://", "https://")):
            # 處理外部 URL
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(img_data)
                response.raise_for_status()
                image_bytes = response.content
        else:
            # 處理純 Base64
            image_bytes = base64.b64decode(img_data)

        # 寫入到 Render 的 Persistent Disk (覆蓋舊檔案)
        # 注意: 這裡使用 asyncio.to_thread 避免阻塞 FastAPI 的主線程
        await asyncio.to_thread(lambda: os.makedirs(os.path.dirname(full_path), exist_ok=True))
        await asyncio.to_thread(lambda: open(full_path, "wb").write(image_bytes))

        return PUBLIC_URL_PREFIX + filename
        
    except Exception as e:
        print(f"Error processing/saving image {filename}: {e}")
        return None


# --- FastAPI 應用實例 ---

@app.on_event("startup")
async def startup_event():
    """服務啟動時檢查並創建磁碟掛載點"""
    os.makedirs(PERSISTENT_STORAGE_PATH, exist_ok=True)


@app.post("/api/generate-and-upload", response_model=Dict[str, Any])
async def generate_and_upload(request: GeneratorRequest):
    """
    呼叫遠端 image-generator 服務，提取圖片並儲存到磁碟。
    """
    
    # --- 1. 呼叫遠端 Image Generator (模擬) ---
    # 這裡假設遠端服務就是您要呼叫的 main.py 的 API 部署實例
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # 假設遠端服務接收與您本服務相同的 JSON 體 (body) 或其他參數
            remote_response = await client.post(
                REMOTE_IMAGE_GENERATOR_URL, 
                json=request.data 
            )
            remote_response.raise_for_status()
            remote_data = remote_response.json()
    except Exception as e:
        # 如果呼叫遠端服務失敗，則直接使用傳入的 JSON 體進行圖片提取
        print(f"Warning: Failed to call remote generator. Using request body for extraction. Error: {e}")
        remote_data = request.data


    # --- 2. 提取圖片字串 ---
    imgs_to_process = find_image_strings(remote_data)
    imgs_to_process = imgs_to_process[:MAX_IMAGES] # 限制最多 4 張

    if not imgs_to_process:
        return JSONResponse(
            status_code=404,
            content={"message": "No image Base64 or URL found in the generator response."}
        )

    # --- 3. 儲存圖片到持久性磁碟 ---
    upload_tasks = [fetch_and_save_image(img, i) for i, img in enumerate(imgs_to_process)]
    uploaded_urls = await asyncio.gather(*upload_tasks)
    
    final_urls = [url for url in uploaded_urls if url]

    return {
        "message": f"Successfully generated and stored {len(final_urls)} images.",
        "uploaded_urls": final_urls
    }


@app.get(PUBLIC_URL_PREFIX + "{filename}")
async def serve_image_from_disk(filename: str):
    """
    公開路由：讓外部使用者存取磁碟上的圖片檔案。
    """
    # 安全性檢查：確保路徑不包含 '..'
    if '..' in filename or not filename.endswith('.png'):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    
    full_path = os.path.join(PERSISTENT_STORAGE_PATH, filename)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    
    # 使用 FileResponse 以優化方式傳輸檔案
    return FileResponse(full_path, media_type="image/png")

# --- 錯誤處理範例 ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "details": str(exc)},
    )

    except Exception as e:
        print(f"[edit_image_api] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")

