# main.py
from fastapi import FastAPI, HTTPException, Request, Query, APIRouter, Form, UploadFile, File
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
import httpx # 確保 httpx 已安裝並導入

# --- 環境變數設定和初始化 ---
# 確保 GOOGLE_API_KEY 是您的環境變數名稱
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")



# --- FastAPI 和 Pydantic 相關匯入 ---

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Gemini API 相關匯入 ---
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Uvicorn 和 Asyncio 相關匯入 (用於 Notebook 啟動) ---



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
def get_full_public_image_url(request: Request, index: int) -> str:
    """
    根據請求物件和索引，組成完整的公開 URL。
    """
    base_url = "https://image-generator-i03j.onrender.com"
    if 0 <= index < len(IMAGE_PATHS):
        filename = IMAGE_PATHS[index]
        # 使用 request.base_url 獲取服務的根 URL (例如 https://image-generator-i03j.onrender.com)
        # 並拼接公開前綴和檔名
        return str(request.base_url).rstrip('/') + PUBLIC_URL_PREFIX + filename
    raise ValueError("Invalid target index.")
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
    
    # 定義所有可能包含圖片字串的鍵
    IMAGE_KEYS = ["image_url", "image", "url", "image_urls", "images", "urls", "results"]
    
    if isinstance(obj, dict):
        for k, value in obj.items():
            
            # 檢查鍵是否是我們預期的圖片鍵
            if k in IMAGE_KEYS:
                
                # 情況 A: 處理單一圖片字串 (例如 "image_url": "base64...")
                if isinstance(value, str) and looks_like_img_url(value):
                    found.append(value)
                
                # 情況 B: 處理圖片陣列 (例如 "image_urls": ["base64...", "http://..."])
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and looks_like_img_url(v):
                            found.append(v)
                        elif isinstance(v, dict):
                            # 遞迴處理陣列內的字典 (以防是巢狀結構)
                            found.extend(find_image_strings(v))
                            
            # 遞迴所有值 (處理巢狀結構)
            elif isinstance(value, (list, dict)):
                found.extend(find_image_strings(value))
                
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
    full_prompt: Optional[str] = None 
    edit_prompt: Optional[str] = None
    image_url: Optional[str] = None 
    image_urls: Optional[List[str]] = None 
    
    # 允許模型接收未在上面明確定義的其他額外鍵值
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
            base64_content = img_data.split(",", 1)[1]
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
    # ❗ 修正點：將詳細的錯誤信息打印出來 ❗
        print(f"--- DISK SAVE ERROR ---")
        print(f"Target Path: {full_path}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("-----------------------")
        return None
# 數據模型 (Pydantic)
class KontextAndImageCreate(BaseModel):
    user_id: Optional[str] = ""
    character_name: Optional[str] = ""
    description: str
    base_prompt: Optional[str] = ""
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
async def get_image_data_for_editing(
    request: Request,
    file: Optional[UploadFile],
    target_index: int
) -> tuple[bytes, str]:
    """
    根據檔案或索引，獲取原始圖片的 bytes 和 MIME Type。
    (這是 edit_image_api 中最核心的圖片獲取邏輯)
    """
    original_image_bytes = None
    image_mime_type = None

    if file and file.filename: 
        # 情況 A: 使用新上傳的檔案
        try:
            original_image_bytes = await file.read()
            image_mime_type = file.content_type or "image/jpeg"
            await file.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"讀取上傳檔案時發生錯誤: {str(e)}")

    else:
        # 情況 B: 使用 target_index 組成的 URL 下載已存圖片
        try:
            url_to_fetch = get_full_public_image_url(request, target_index)
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url_to_fetch)
                response.raise_for_status() 
                original_image_bytes = response.content
                image_mime_type = response.headers.get("Content-Type", "image/png")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"無法從已儲存的圖片 (Index {target_index}) 下載圖片。請確認檔案是否存在。錯誤：{str(e)}"
            )
            
    if not original_image_bytes:
        raise HTTPException(status_code=500, detail="無法獲取圖片數據，請檢查輸入。")

    return original_image_bytes, image_mime_type
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
#@app.post("/edit_image")
#async def edit_image_api(
#    edit_prompt: str = Form(...),
#    file: UploadFile = File(...)
#):
#    """
#    呼叫 gemini_image_editing 進行圖片修改。
#    前端上傳圖片與提示詞即可，例如：
#    FormData:
#      - edit_prompt: "讓畫面更明亮，保持手繪質感"
#      - file: <image>
#    """

 #   try:
 #       # 讀取上傳的圖片 bytes
 #       original_image_bytes = await file.read()
 #       image_mime_type = file.content_type or "image/jpeg"

        # 呼叫你原本的函式
 #       edited_image_data_url = gemini_image_editing(
 #           edit_prompt=edit_prompt,
 #           original_image_bytes=original_image_bytes,
 #           image_mime_type=image_mime_type
 #       )

  #      if not edited_image_data_url:
   #         raise HTTPException(
    #            status_code=500,
     #           detail="Gemini 沒有返回圖片資料，請檢查模型權限或提示詞。"
      #      )

       # return {
        #    "edit_prompt": edit_prompt,
         #   "image_url": edited_image_data_url
        #}

   # except Exception as e:
   #     print(f"[edit_image_api] Error: {e}")
   #     raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")



# 假設所有輔助函式 (get_full_public_image_url, gemini_image_editing) 已經定義在其他地方
@app.post("/edit_image", response_model=Dict[str, Any])
async def edit_image_api(
    request: Request,
    edit_prompt: str = Form(...),
    
    # target_index 設為必填 Query 參數
    target_index: int = Query(..., ge=0, le=3,
                              description="目標圖片索引 (0=001.png, 1=002.png, ..., 3=004.png)"),
                              
    # file 設為可選 File 參數
    file: Optional[UploadFile] = File(None)
):
    """
    進行圖片修改。若傳入檔案，則使用新檔案；若未傳入，則使用 target_index 指定的已存圖片。
    """
    # 變數初始化 (解決 name '...' is not defined 錯誤)
    original_image_bytes = None
    image_mime_type = None
    edited_image_data_url = None # 初始化最終結果變數

    # --- 1. 檢查並處理上傳檔案 (優先級最高) ---
    
    # 檢查 file 是否存在且有檔名 (file.filename 檢查可以排除空字串的上傳，但仍需客戶端配合)
    if file and file.filename:
        # 情況 A: 使用新上傳的檔案
        try:
            original_image_bytes = await file.read()
            image_mime_type = file.content_type or "image/jpeg"
            await file.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"讀取上傳檔案時發生錯誤: {str(e)}")

    # --- 2. 處理下載已儲存的圖片 (沒有上傳新檔案時的預設邏輯) ---
    else:
        # 情況 B: 使用 target_index 組成的 URL 下載已存圖片
        try:
            # 在後端組成完整的 URL
            url_to_fetch = get_full_public_image_url(request, target_index)
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url_to_fetch)
                response.raise_for_status() # 檢查 4xx/5xx 錯誤
                
                original_image_bytes = response.content
                image_mime_type = response.headers.get("Content-Type", "image/png")

        except ValueError as ve:
            # 處理 target_index 範圍錯誤
            raise HTTPException(status_code=400, detail=f"圖片索引錯誤: {str(ve)}")
        except Exception as e:
            # 下載失敗的錯誤 (例如 Render Disk 上的檔案不存在)
            raise HTTPException(
                status_code=500,
                detail=f"無法從已儲存的圖片 (Index {target_index}) 下載圖片。請確認檔案是否存在。錯誤：{str(e)}"
            )
            
    # --- 3. 呼叫圖片編輯邏輯 ---
    if not original_image_bytes:
        # 如果走到這裡，表示所有圖片獲取途徑都失敗了
        raise HTTPException(status_code=500, detail="無法獲取圖片數據，請檢查輸入或圖片是否存在。")
        
    try:
        # ❗ 假設 gemini_image_editing 是一個同步函式 ❗
        edited_image_data_url = gemini_image_editing(
            edit_prompt=edit_prompt,
            original_image_bytes=original_image_bytes,
            image_mime_type=image_mime_type
        )
    except Exception as e:
        # 捕捉 gemini_image_editing 內部錯誤
        raise HTTPException(status_code=500, detail=f"圖片編輯處理失敗: {str(e)}")

    # --- 4. 最終返回 ---
    if not edited_image_data_url:
        raise HTTPException(status_code=500, detail="編輯模型沒有返回有效的圖片數據。")

    return {
        "edit_prompt": edit_prompt,
        "image_url": edited_image_data_url
    }
    
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
    uploaded_urls = await save_image_to_disk(imgs_to_process_ , target_index) 

    #uploaded_urls = asyncio.gather(*upload_tasks) #如只處理一張圖片不需要了
    if not uploaded_urls:
        raise HTTPException(status_code=500, detail="Failed to save image to disk.")
    #final_urls = [url for url in uploaded_urls if url]
    final_urls = [uploaded_urls]
    return {
            "message": f"Successfully stored 1 image to persistent disk (Index {target_index}).",
            "uploaded_urls": final_urls
        }

@app.post("/edit_image_store", response_model=Dict[str, Any])
async def edit_image_and_store(
    request: Request,
    edit_prompt: str = Form(...),
    target_index: int = Query(0, ge=0, le=3, 
                              description="目標圖片索引 (0-3)，用於輸入和儲存的檔案編號"),
    file: Optional[UploadFile] = File(None)
):
    """
    執行圖片編輯，並將編輯後的 Base64 圖片儲存到 Render Disk 上的目標索引位置。
    """
    
    # 步驟 A: 獲取原始圖片數據 (使用 edit_image 的邏輯)
    original_bytes, mime_type = await get_image_data_for_editing(request, file, target_index)

    # 步驟 B: 呼叫圖片編輯邏輯
    try:
        # 假設 edited_image_data_url 是 data:image/png;base64,... 格式的字串
        edited_image_data_url = gemini_image_editing(
            edit_prompt=edit_prompt,
            original_image_bytes=original_bytes,
            image_mime_type=mime_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片編輯處理失敗: {str(e)}")

    if not edited_image_data_url:
        raise HTTPException(status_code=500, detail="編輯模型沒有返回有效的圖片數據。")

    # 步驟 C: 儲存編輯後的圖片 (使用 store_generated_images 的邏輯)
    
    # 儲存邏輯的輸入是 Base64 字串，所以我們將編輯結果作為輸入
    image_data_to_store = edited_image_data_url 
    
    # 傳入 target_index 確保覆蓋目標檔案 (001.png 到 004.png)
    stored_url = await save_image_to_disk(image_data_to_store, target_index) 

    if not stored_url:
        raise HTTPException(status_code=500, detail="Failed to save edited image to persistent disk.")

    # 步驟 D: 最終回傳
    final_urls = [stored_url]

    return {
        "message": f"Successfully edited and stored image to disk (Index {target_index}).",
        "edit_prompt": edit_prompt,
        "image_url": edited_image_data_url, # 編輯後的 Base64 Data URL
        "uploaded_urls": final_urls          # 編輯後圖片的公開存取 URL
    }

# ... (其他導入和常數保持不變)

@app.post("/generate_image_store", response_model=Dict[str, Any])
async def generate_image_store(
    payload: KontextAndImageCreate,
    # ❗ 修正點 1: 新增起始索引參數 ❗
    target_index: int = Query(0, ge=0, le=(MAX_IMAGES - 1), 
                                    description="生成的圖片開始儲存的索引 (0=001.png, 1=002.png)")
):
    """
    執行圖片生成，並將生成的圖片儲存到 Render 磁碟上，從 target_start_index 開始覆蓋。
    """
    ##
        # 組合提示詞
    base_prompt = payload.base_prompt if payload.base_prompt else ""
    full_prompt = f"{payload.description}. {base_prompt}"
    
    # 獲取 Base64 Data URLs
    images = gemini_image_generation(full_prompt, count=payload.image_count)

    if not images:
        raise HTTPException(
            status_code=500,
            detail="Gemini generation failed or no image data returned."
        )

    try:
        # 儲存邏輯的輸入是 Base64 字串，所以我們將編輯結果作為輸入
        image_data_to_store = images[0]
        
        # 傳入 target_index 確保覆蓋目標檔案 (001.png 到 004.png)
        stored_url = await save_image_to_disk(image_data_to_store, target_index) 
    
        if not stored_url:
            raise HTTPException(status_code=500, detail="Failed to save edited image to persistent disk.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片儲存到磁碟失敗: {str(e)}")

    final_urls = [stored_url]        

    if not final_urls:
         raise HTTPException(status_code=500, detail="圖片已生成，但儲存到磁碟全部失敗。")
         
    # --- 3. 最終回傳 ---
    return {
        "message": f"Successfully generated and stored {len(final_urls)} images, starting from index {target_start_index}.",
        "full_prompt": full_prompt,
        "image_urls": images,      
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
