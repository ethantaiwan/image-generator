# main.py
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import json, os
import base64
import uuid
from typing import Any, Dict, List, Union, Optional
import re
import io
import asyncio
# --- 環境變數設定和初始化 ---
# 確保 GOOGLE_API_KEY 是您的環境變數名稱
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")



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
PERSISTENT_STORAGE_PATH = "/var/data" 
MAX_IMAGES = 4
IMAGE_PATHS = [f"00{i}.png" for i in range(1, MAX_IMAGES + 1)]
PUBLIC_URL_PREFIX = "/image-uploads/temp/"

# --- 假設遠端服務的 URL ---
# 請將這裡替換成您實際部署 image-generator 的 API 地址
REMOTE_IMAGE_GENERATOR_URL = "https://https://image-generator-i03j.onrender.com/api/image-generator" 


# --- 輔助函式：JSON 圖片字串提取 (根據您的要求) ---

def looks_like_img_url(s: str) -> bool:
    """粗略判斷字串是否為圖片連結或 Base64 字串"""
    s = s.strip()
    return (
        s.startswith("data:image/") or
        s.startswith("http://") or s.startswith("https://") or
        # 僅用於判斷純 Base64 字串，但強烈建議使用 data:image/ 前綴
        (re.fullmatch(r"[A-Za-z0-9+/=\s]+", s or "") and len(s) > 100) 
    )

def find_image_strings(obj: Union[Dict, List]) -> List[str]:
    """遞迴地在 JSON 結構中尋找圖片連結或 Base64 字串"""
    found = []
    if isinstance(obj, dict):
        # 查找您需要的鍵: "image_url" (單數) 和 "image_urls" (複數)
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


# --- Pydantic 模型用於請求 Body (接收您的生成 JSON 輸出) ---
class GeneratorOutput(BaseModel):
    """用於接收您的生成 API 輸出的 JSON 結構"""
    full_prompt: str
    image_urls: List[str]  # 這是您提取圖片的關鍵鍵
    # 也可以加上其他您可能傳入的鍵，例如:
    # edit_prompt: Optional[str] = None 
    # data: Optional[Dict[str, Any]] = None # 如果您仍需處理外層 data 鍵
    
    # 允許模型接收未在上面明確定義的其他額外鍵值 (extra fields)
    class Config:
        extra = "allow"
# --- 圖片儲存邏輯 ---

async def save_image_to_disk(img_data: str, index: int) -> Union[str, None]:
    """將 Base64 或 URL 圖片儲存到持久性磁碟"""
    filename = IMAGE_PATHS[index]
    full_path = os.path.join(PERSISTENT_STORAGE_PATH, filename)
    
    try:
        if img_data.startswith("data:image/"):
            # 處理 Base64 (移除 data:image/png;base64, 前綴)
            #base64_content = img_data.split(",", 1)[1]
            base64_content = img_data.imgs_str.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_content)
        elif img_data.startswith(("http://", "https://")):
            # 處理外部 URL (由於您希望精簡，這裡將會返回錯誤，因為我們移除了 httpx)
            raise ValueError("External URL processing is disabled in this simplified service.")
        else:
            # 處理純 Base64
            image_bytes = base64.b64decode(img_data)

        # 寫入到 Render 的 Persistent Disk (覆蓋舊檔案)
        # 使用 asyncio.to_thread 避免阻塞
        await asyncio.to_thread(lambda: os.makedirs(os.path.dirname(full_path), exist_ok=True))
        await asyncio.to_thread(lambda: open(full_path, "wb").write(image_bytes))

        return PUBLIC_URL_PREFIX + filename
        
    except Exception as e:
        # 捕捉所有儲存或解碼錯誤
        print(f"Error processing/saving image {filename}: {e}")
        return None
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
def gemini_image_generation(prompt: str,count: int = 1) -> List[str]:
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
@app.on_event("startup")
async def startup_event():
    """服務啟動時檢查並創建磁碟掛載點"""
    os.makedirs(PERSISTENT_STORAGE_PATH, exist_ok=True)
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

    except Exception as e:
        print(f"[edit_image_api] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")


@app.post("/api/store-generated-images", response_model=Dict[str, Any])
async def store_generated_images(
    request_body: GeneratorOutput,
    # ❗ 修正點 1: 新增 Query 參數來決定覆蓋的檔案編號 ❗
    target_index: int = Query(0, ge=0, le=(MAX_IMAGES - 1), 
                              description="目標檔案索引 (0=001.png, 1=002.png, ..., 3=004.png)")
):
    """
    接收生成 API 的輸出 JSON，提取 Base64 圖片並儲存到 Render 磁碟。
    """
    
    # 這裡直接使用傳入的 JSON 體進行圖片提取
    json_data = request_body.model_dump()
    # --- 提取圖片字串 ---
    imgs_to_process = find_image_strings(json_data)
    
    # 限制最多 4 張，並覆蓋固定的檔名 001.png 到 004.png
    imgs_to_process_ = imgs_to_process[0] 

    if not imgs_to_process_:
        return JSONResponse(
            status_code=404,
            content={"message": "No image Base64 or URL found in the provided JSON."}
        )

    # --- 儲存圖片到持久性磁碟 ---
    #upload_tasks = [save_image_to_disk(img, i) for i, img in enumerate(imgs_to_process)]
    upload_tasks = await save_image_to_disk(imgs_to_process_ , target_index) 

    uploaded_urls = asyncio.gather(*upload_tasks) #如只處理一張圖片不需要了
    if not uploaded_urls:
        raise HTTPException(status_code=500, detail="Failed to save image to disk.")
    #final_urls = [url for url in uploaded_urls if url]
    final_urls = [uploaded_urls]
    return {
            "message": f"Successfully stored 1 image to persistent disk (Index {target_index}).",
            "uploaded_urls": final_urls
        }


@app.get(PUBLIC_URL_PREFIX + "{filename}")
async def serve_image_from_disk(filename: str):
    """
    公開路由：讓外部使用者存取磁碟上的圖片檔案 (e.g., .../lovable-uploads/temp/001.png)
    """
    # 安全性檢查：確保路徑安全且是 PNG
    if '..' in filename or not filename.endswith('.png') or filename not in IMAGE_PATHS:
        raise HTTPException(status_code=400, detail="Invalid filename or file type.")
    
    full_path = os.path.join(PERSISTENT_STORAGE_PATH, filename)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found in persistent disk.")
    
    # 使用 FileResponse 傳輸檔案
    return FileResponse(full_path, media_type="image/png")


# --- 錯誤處理範例 ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "details": str(exc)},
    )
