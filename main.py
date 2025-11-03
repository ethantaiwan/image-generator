# main.py
from fastapi import FastAPI, HTTPException
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
from typing import List, Optional

# --- FastAPI 和 Pydantic 相關匯入 ---
from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Gemini API 相關匯入 ---
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Uvicorn 和 Asyncio 相關匯入 (用於 Notebook 啟動) ---
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import asyncio 

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
MODEL_NAME = "gemini-2.5-flash-image" 

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
    model = "gemini-2.5-flash-image" 

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
    model = "gemini-2.5-flash-image" 

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

    except Exception as e:
        print(f"[edit_image_api] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")
