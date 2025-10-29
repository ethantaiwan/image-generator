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
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

# 圖像生成模型
# 注意：使用 gemini-2.5-flash 來生成圖像時，它會自動呼叫底層的 Imagen 模型。
# 您的原始程式碼中 model = GenerativeModel(model_name="gemini-2.5-flash-image-preview") 
# 或許會因 SDK 版本而異。這裡我們保留您的寫法，但請確認您的 google-genai 版本支持。
# 最標準的寫法是使用 DALL-E/Imagen 專屬的 API，但如果您的目標是使用 gemini-2.5-flash 驅動，則保持原樣。
model = GenerativeModel(model_name="gemini-2.5-flash-image-preview")  


app = FastAPI()

# 檔案和路徑設定
# 在 Render 等無狀態環境中，不建議使用本地文件來儲存狀態 (kontexts.json)
# 並且本地儲存的圖片 (images/...) 會在容器重啟時丟失。
# **重要提醒**: 
# - 如果需要持久化狀態，請使用 **PostgreSQL** 或 **Redis** 等資料庫。
# - 如果需要儲存圖片，請使用 **Google Cloud Storage** 或 **AWS S3** 等雲端儲存服務。
# 為了讓程式碼能夠運行，我們將 `kontexts.json` 的功能移除，並將圖片輸出改為 **base64 data URL**，
# 這樣 API 就可以直接返回圖片資料，無需儲存本地檔案。
# 同時，我們也移除了用到本地檔案的 `/create_kontext_and_generate` 中的儲存邏輯。

KONTEXT_FILE = "kontexts.json"

class KontextAndImageCreate(BaseModel):
    user_id: str
    character_name: str
    description: str
    base_prompt: Optional[str] = None
    image_count: int = 3

class GenerateRequest(BaseModel):
    user_id: str
    character_name: str
    prompt: str
    image_count: int = 1

class ImageBatchResponse(BaseModel):
    full_prompt: str
    image_urls: List[str]

class ImageResponse(BaseModel):
    full_prompt: str
    image_url: str

# 移除 load_kontexts 和 save_kontexts 函數 (因為不適合無狀態環境)

# --- 圖片生成函數優化 ---
# 專注於生成並回傳 Base64 Data URL
def gemini_image_generation(prompt: str, count: int = 1) -> List[str]:
    """使用 Gemini API 生成圖片並回傳 Base64 Data URL 列表"""
    try:
        # 為了獲得圖片輸出，我們必須使用 model.generate_content 並檢查 parts
        response = client.models.generate_content(
            model='gemini-2.5-flash-image-preview',
            contents=[prompt],
            config={
                # 這裡設定 max_output_tokens=0 或 1，表示不期望有文字輸出
                # 但是，為了確保獲得圖像，您需要依賴 model 的能力。
                # 許多 SDK 版本會直接接受文字提示，然後在 part 中返回圖像數據。
                # 讓我們將其保留為一個簡單的 generate_content 呼叫
                # 如果 model="gemini-2.5-flash-image-preview" 能夠返回圖像，它會在 part 中。
            }
        )
        
        images_data_urls = []
        for part in response.parts:
           if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
                img_bytes = part.inline_data.data
                if isinstance(img_bytes, str):
                    # 某些情況下，數據是 base64 編碼的字串
                    img_bytes = base64.b64decode(img_bytes)
                
                encoded = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = part.inline_data.mime_type or "image/jpeg"
                data_url = f"data:{mime_type};base64,{encoded}"
                images_data_urls.append(data_url)

        return images_data_urls[:count]
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # 如果 API 返回的 response 沒有圖像數據，這裡會返回空列表
        return []

# 移除 gemini_image 函數，使用優化後的 gemini_image_generation

# --- API Endpoint 調整 ---
@app.post("/create_kontext_and_generate", response_model=ImageBatchResponse)
def create_kontext_and_generate(payload: KontextAndImageCreate):
    
    # 由於我們移除了文件持久化，這裡只是執行生成邏輯
    full_prompt = f"{payload.description}. {payload.base_prompt}"
    
    # 獲取 Base64 Data URLs
    images = gemini_image_generation(full_prompt, count=payload.image_count)

    if not images:
        raise HTTPException(status_code=500, detail="Gemini generation failed or no image data returned.")

    # **重要提醒**: 
    # 因為沒有持久化機制，這裡無法像原先一樣儲存 base_image 供後續使用。
    # 為了演示功能，我們直接回傳生成的圖片 URLs。
    
    # 如果您需要儲存 base_image，您必須將其上傳到雲端儲存服務（如 GCS/S3），
    # 並將返回的 URL 儲存到雲端資料庫（如 PostgreSQL/MongoDB）。

    return ImageBatchResponse(full_prompt=full_prompt, image_urls=images)

# 您註解掉的 generate_image endpoint 由於涉及本地文件讀寫和 kontexts 儲存，
# 在 Render 環境中需要完全重寫為使用雲端資源，這裡暫時將其保留為註解。
