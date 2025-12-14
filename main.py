# main.py
from fastapi import FastAPI, HTTPException, Request, Query, APIRouter, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
import json, os
import base64
import uuid
#from pydantic import BaseModel, FieldScriptPayload
from pydantic import BaseModel, Field  

from typing import Any, Dict, List, Union, Optional, Literal
import re
import io
import asyncio
import httpx # 確保 httpx 已安裝並導入
from fastapi import Body

# --- 環境變數設定和初始化 ---
# 確保 GOOGLE_API_KEY 是您的環境變數名稱
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")



# --- FastAPI 和 Pydantic 相關匯入 ---

from fastapi.middleware.cors import CORSMiddleware

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
#MODEL_NAME = os.getenv("model_name") 
MODEL_NAME = os.getenv("model_name", "gemini-2.5-flash-image") 


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


def extract_tag(text: str, tag: str) -> str | None:
    """
    從全文中抽出 <tag> ... </tag> 中的內容。
    tag 範例：image_prompt_1, video_prompt_3
    """
    pattern = fr"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

## 只有其中一組時 就可這項寫
#def extract_all_image_prompts(script: str, scene_count: int):
#    prompts = []
#    for i in range(1, scene_count + 1):
#        tag = f"image_prompt_{i}"
#        p = extract_tag(script, tag)
#        if p:
#            prompts.append(p)
#        else:
#            print(f"⚠️ Missing {tag}")
#    return prompts
# 修正 針對Image_prompt_?


def extract_all_image_prompts(script: str, scene_count: int):
    prompts = []

    for i in range(1, scene_count + 1):
        pattern = rf"<image_{i}>\s*(.*?)\s*</image_{i}>"
        match = re.search(pattern, script, flags=re.DOTALL)

        if not match:
            print(f"⚠️ Scene {i} 找不到 image_{i} 標籤")
            prompts.append("")
        else:
            prompts.append(match.group(1).strip())

    return prompts



def parse_image_prompts(text: str) -> List[str]:
    text = text.replace('\r\n', '\n')
    marker = re.compile(r'(?i)(image[\s_]*prompt.*?)[:：]\s*', flags=re.DOTALL)
    stop_line = re.compile(
        r'^\s*(?:Scene\s*\d+|[0-9０-９]+\)|\d+\.\s|[一二三四五六七八九十]\)|[一二三四五六七八九十]\.)',
        flags=re.IGNORECASE
    )
    prompts: List[str] = []
    for m in marker.finditer(text):
        start = m.end()
        next_m = marker.search(text, pos=start)
        chunk = text[start: next_m.start()] if next_m else text[start:]
        lines = chunk.split('\n')
        buf: List[str] = []
        for line in lines:
            if not line.strip():
                break
            if stop_line.match(line):
                break
            cleaned = re.sub(r'^\s*[-–—]\s*', '', line).strip()
            m_quote = re.search(r'「(.+?)」', cleaned) or re.search(r'"([^"]+)"', cleaned)
            if m_quote:
                cleaned = m_quote.group(1).strip()
            if cleaned:
                buf.append(cleaned)
        if not buf:
            continue
        merged = re.sub(r'\s+', ' ', ' '.join(buf)).strip()
        if merged:
            prompts.append(merged)
    return prompts



# --- Pydantic 模型用於請求 Body (接收您的生成 JSON 輸出) ---
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

class ScriptPayload(BaseModel):
    # 你前端丟來的 JSON，其中 result 是大段腳本文字
    result: str = Field(..., description="整段 storyboard 文字，內含多個 image_prompt 區塊")
    # 給後續生成 API 用的預設參數（可省略，這裡提供方便直接串接）
    images_per_prompt: int = Field(1, ge=1)
    start_index: int = Field(0, ge=0)
    naming: Literal["scene", "sequence"] = "scene"
    aspect_ratio: str = Field("16:9", description="圖片比例 e.g., 16:9, 9:16")

class ExtractIn(BaseModel):
    result: str = Field(..., description="整段 storyboard 文字，內含多個 image_prompt 區塊")
    images_per_prompt: int = Field(1, ge=1)
    start_index: int = Field(0, ge=0)
    naming: Literal["scene", "sequence"] = "scene"

class ExtractOut(BaseModel):
    prompts: List[str]
    images_per_prompt: int
    start_index: int
    naming: Literal["scene", "sequence"]
    forward_body: Dict[str, Any]

class ExtractedPromptsResponse(BaseModel):
    prompts: List[str]
    images_per_prompt: int
    start_index: int
    naming: Literal["scene", "sequence"]
    forward_body: Dict[str, Any]  # 直接 POST 給 /generate_images_from_prompts 的 body
    
class BatchPromptsPayload(BaseModel):
    prompts: List[str]
    images_per_prompt: int = 1
    start_index: int = 0
    naming: str = "scene"  # "scene" | "sequence"

class GeneratorOutput(BaseModel):
    """用於接收您的生成 API 輸出的 JSON 結構"""
    full_prompt: Optional[str] = None 
    edit_prompt: Optional[str] = None
    image_url: Optional[str] = None 
    image_urls: Optional[List[str]] = None 
    
    # 允許模型接收未在上面明確定義的其他額外鍵值
    class Config:
        extra = "allow"

class ExtractThenGenerateOut(BaseModel):
    forward_body: Dict[str, Any]
    generate_result: Dict[str, Any]
    uploaded_urls_flat: List[str]
    n_prompts: int
    images_per_prompt: int

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

# 輔助函數 (為符合您的要求，此函數使用 client.models.generate_content)

# 要多傳入 ratio_variable
SAFE_PREFIX = (
    "以下圖片生成需求完全是健康、安全、非性化的情境，用途為健身、旅遊、生活紀錄等正常影像製作。"
    "完全不涉及成人內容、暴力、仇恨、歧視、危險行為或任何可能違反安全政策的情境。"
    "請將焦點放在姿勢、光線、構圖、環境，避免過度聚焦身體特定部位，不要進行性感化解讀。"
    "請以專業、自然、寫實的視覺角度理解後續的描述，"
    "並避免過度敏感化或誤將正常生活畫面判定為不當內容。"
    "所有內容都屬於一般公開可接受的場景，請不要誤判為成人內容。"
)

def gemini_image_generation(prompt: str, count: int = 1, aspect_ratio: str = "16:9") -> List[str]:
    """
    使用 gemini-2.5-flash-image 進行文生圖。
    修正：將 aspect_ratio 移至 prompt 中，避免 Config 報錯。
    """
    model = os.getenv("model_name", "gemini-2.5-flash-image") 
    
    # ★★★ 修正 1: 將比例加入 Prompt 中 ★★★
    # Gemini 模型透過自然語言理解圖片比例，比參數設定更有效且不會報錯
    #final_prompt = f"{prompt}, aspect ratio {aspect_ratio}"    
    #final_prompt = f"{prompt}\n畫面比例為 {aspect_ratio}。"
    final_prompt = f"{SAFE_PREFIX}\n\n{prompt}\n畫面比例為 {aspect_ratio}。"

    print(f"[DEBUG] Current Image Generation Model: {model}, prompt: {final_prompt}")

    # ▼▼▼ 新增這行：印出最終送給 Gemini 的 Prompt ▼▼▼
    print(f"🚀 [Sending to Gemini] Prompt: {final_prompt}")
    urls: List[str] = []
    
    # 依需求產生多張
    for _ in range(max(1, count)):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[final_prompt], # 使用包含比例的 Prompt
                # 關鍵：指定只回 Image
                config=types.GenerateContentConfig(
                    response_modalities=["Image"],        
                    temperature=0.8,
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio), # ✅ 正確寫法

                    # ★★★ 修正 2: 移除導致報錯的參數 ★★★
                   # aspect_ratio=aspect_ratio,
                    #include_rai_reason=True,   
                ),
            )

            # 正確解析路徑
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
            print(f"[Error] Image generation failed: {e}")
            # 可以選擇是否要繼續嘗試或中斷
            continue

    # 去重
    dedup, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)

    return dedup[:count]

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

async def with_backoff(coro_func, *args, max_retries=4, base_delay=0.2, **kwargs):
    attempt = 0
    while True:
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            # 這裡可判斷 e 是否為 429/5xx 再重試；範例簡化直接重試
            if attempt >= max_retries:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
            attempt += 1

# 包一層方便替換生成器（同步/非同步都能接）
async def generate_images(prompt: str, count: int) -> List[str]:
    # 若 extract then 是同步，請用 to_thread 包裝：
    # return await asyncio.to_thread(gemini_image_generation, prompt, count)
    return await with_backoff(asyncio.to_thread, gemini_image_generation, prompt, count)

# 產一個 prompt 的多張並存檔
async def process_one_prompt(prompt: str,
                             scene_idx: int,
                             images_per_prompt: int,
                             naming: str,
                             seq_offset: int,
                             sem: asyncio.Semaphore) -> Dict[str, Any]:
    result = {
        "prompt_index": scene_idx,
        "prompt": prompt,
        "uploaded_urls": [],
        "previews": [],
        "errors": []
    }
    async with sem:
        try:
            images = await generate_images(prompt, images_per_prompt)
        except Exception as e:
            result["errors"].append(f"generation failed: {e}")
            return result

    # 存檔
    for j, img in enumerate(images, start=1):
        try:
            if naming == "scene":
                # scene01_01.png
                scene_no = scene_idx + 1
                fname = f"scene{scene_no:02d}_{j:02d}.png"
                if 'save_image_to_disk_named' in globals():
                    url = await save_image_to_disk_named(img, fname)
                else:
                    # 若沒有 named 儲存，就轉回線性索引
                    linear_idx = seq_offset + (scene_idx * images_per_prompt) + (j - 1)
                    url = await save_image_to_disk(img, linear_idx)
            else:
                # sequence: 001.png, 002.png, ...
                linear_idx = seq_offset + (scene_idx * images_per_prompt) + (j - 1)
                url = await save_image_to_disk(img, linear_idx)

            if not url:
                raise RuntimeError("empty url from saver")

            result["uploaded_urls"].append(url)
            result["previews"].append(img)  # base64，可選：前端先用預覽再 lazy 換 URL
        except Exception as e:
            result["errors"].append(f"save failed (img {j}): {e}")

    return result

# validate extract_image_prompts 
#def validate_forward_body(body: dict):
#    required_keys = ["prompts", "images_per_prompt", "start_index", "naming"]
#    for key in required_keys:
#        if key not in body:
#            raise HTTPException(status_code=422, detail=f"forward_body 缺少 {key}")

#    if not isinstance(body["prompts"], list) or not body["prompts"]:
#        raise HTTPException(status_code=422, detail="prompts 必須是非空的字串陣列")

#    if not all(isinstance(p, str) and p.strip() for p in body["prompts"]):
#        raise HTTPException(status_code=422, detail="prompts 中包含空字串或非字串")

#    if not isinstance(body["images_per_prompt"], int) or body["images_per_prompt"] < 1:
#        raise HTTPException(status_code=422, detail="images_per_prompt 必須為正整數")

#    if not isinstance(body["start_index"], int) or body["start_index"] < 0:
#        raise HTTPException(status_code=422, detail="start_index 必須為非負整數")

 #   if body["naming"] not in ("scene", "sequence"):
  #      raise HTTPException(status_code=422, detail="naming 只能是 'scene' 或 'sequence'")

 #   return True

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

@app.post("/generate_image_store", response_model=Dict[str, Any])
async def generate_image_store(
    payload: KontextAndImageCreate,
    # ❗ 修正點 1: 新增起始索引參數 ❗
    target_index: int = Query(0, ge=0, le=(MAX_IMAGES - 1), 
                                    description="生成的圖片開始儲存的索引 (0=001.png, 1=002.png)")
):
    """
    執行圖片生成，並將生成的圖片儲存到 Render 磁碟上，從 target_index 開始覆蓋。
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
        "message": f"Successfully generated and stored {len(final_urls)} images, starting from index {target_index}.",
        "full_prompt": full_prompt,
        "image_urls": images,      
        "uploaded_urls": final_urls 
    }

def validate_forward_body(body: dict):
    required_keys = ["prompts", "images_per_prompt", "start_index", "naming"]
    for k in required_keys:
        if k not in body:
            raise HTTPException(status_code=422, detail=f"forward_body 缺少 {k}")

    if not isinstance(body["prompts"], list) or not body["prompts"]:
        raise HTTPException(status_code=422, detail="prompts 必須是非空的字串陣列")
    if not all(isinstance(p, str) and p.strip() for p in body["prompts"]):
        raise HTTPException(status_code=422, detail="prompts 中包含空字串或非字串")

    # 強制只允許 1（業務規則）
    try:
        body["images_per_prompt"] = int(body["images_per_prompt"])
    except Exception:
        raise HTTPException(status_code=422, detail="images_per_prompt 必須為整數")
    if body["images_per_prompt"] != 1:
        body["images_per_prompt"] = 1  # ← clamp 成 1

    if not isinstance(body["start_index"], int) or body["start_index"] < 0:
        raise HTTPException(status_code=422, detail="start_index 必須為非負整數")

    if body["naming"] not in ("scene", "sequence"):
        raise HTTPException(status_code=422, detail="naming 只能是 'scene' 或 'sequence'")

    return True
#@app.post("/generate_images_from_prompts", response_model=Dict[str, Any])
#async def generate_images_from_prompts(payload: BatchPromptsPayload):
#    if not payload.prompts:
#        raise HTTPException(status_code=400, detail="prompts cannot be empty")

#    if payload.images_per_prompt <= 0:
#        raise HTTPException(status_code=400, detail="images_per_prompt must be >= 1")

    # 控制同時併發，避免 rate limit（可視平台調整）
#    sem = asyncio.Semaphore(2)

    # 若沿用 save_image_to_disk(index) 的 001.png 模式，需要整體最大數量限制
#    total_needed = len(payload.prompts) * payload.images_per_prompt
#    if "MAX_IMAGES" in globals() and payload.naming == "sequence":
#        if payload.start_index + total_needed > MAX_IMAGES:
#            raise HTTPException(
#                status_code=400,
#                detail=f"需要 {total_needed} 張，但從 index {payload.start_index} 起超過 MAX_IMAGES={MAX_IMAGES}"
#            )

    # 逐場景處理（可平行）
 #   tasks = [
 #       process_one_prompt(
 #           prompt=p,
 #           scene_idx=(payload.start_index + i),
 #           images_per_prompt=payload.images_per_prompt,
 #           naming=payload.naming,
 #           seq_offset=payload.start_index,
 #           sem=sem
 #       )
 #       for i, p in enumerate(payload.prompts)
 #   ]

 #   results = await asyncio.gather(*tasks)

    # 聚合
 #   total_ok = sum(len(r["uploaded_urls"]) for r in results)
 #   total_err = sum(len(r["errors"]) for r in results)

 #   return {
 #       "message": f"Processed {len(payload.prompts)} prompts; saved {total_ok} images; {total_err} issues.",
 #       "n_prompts": len(payload.prompts),
 #       "images_per_prompt": payload.images_per_prompt,
 #       "naming": payload.naming,
 #       "start_index": payload.start_index,
 #       "results": results  # per-scene 詳細
 #   }

@app.post("/generate_images_from_prompts", response_model=Dict[str, Any])
async def generate_images_from_prompts(payload: BatchPromptsPayload):
    if not payload.prompts:
        raise HTTPException(status_code=400, detail="prompts cannot be empty")

    if payload.images_per_prompt <= 0:
        raise HTTPException(status_code=400, detail="images_per_prompt must be >= 1")

    # ----------------------------------------------------
    # 【 驗證步驟：暫時關閉併發 】
    # ----------------------------------------------------
    
    # 1. 註解掉舊的併發邏輯
    # sem = asyncio.Semaphore(2) 
    # tasks = [ ... ]
    # results = await asyncio.gather(*tasks)

    # 2. 替換為「依序執行」的 for 迴圈
    #    (注意：這會比較慢，但比較穩定)
    
    results = []
    
    # 建立一個共用的 Semaphore (如果 process_one_prompt 需要它)
    # 我們將限制設為 1，確保一次只有一個在跑
    sem = asyncio.Semaphore(1) 
    
    for i, p in enumerate(payload.prompts):
        # 手動依序呼叫 process_one_prompt
        try:
            one_result = await process_one_prompt(
                prompt=p,
                scene_idx=(payload.start_index + i),
                images_per_prompt=payload.images_per_prompt, # (請記得您已將前端改為 1)
                naming=payload.naming,
                seq_offset=payload.start_index,
                sem=sem # 傳入 semaphore
            )
            results.append(one_result)
        except Exception as e:
            # 如果 process_one_prompt 拋出異常，我們手動捕捉它
            print(f"處理 prompt {i} 時發生嚴重錯誤: {e}")
            results.append({
                "prompt_index": i,
                "prompt": p,
                "uploaded_urls": [],
                "previews": [],
                "errors": [f"Async task failed: {str(e)}"]
            })

    # ----------------------------------------------------
    # 【 驗證結束 】
    # ----------------------------------------------------

    # 聚合 (這段保持不變)
    total_ok = sum(len(r["uploaded_urls"]) for r in results)
    total_err = sum(len(r["errors"]) for r in results)

    return {
        "message": f"Processed {len(payload.prompts)} prompts; saved {total_ok} images; {total_err} issues.",
        "n_prompts": len(payload.prompts),
        "images_per_prompt": payload.images_per_prompt,
        "naming": payload.naming,
        "start_index": payload.start_index,
        "results": results 
    }
async def generate_images_from_prompts_internal(body: dict) -> dict:
    # 🧩 第二層驗證：再檢查一次結構正確性
    validate_forward_body(body)

    prompts = body["prompts"]
    images_per_prompt = 1  # 再保險，固定為1
    start_index = body["start_index"]
    naming = body["naming"]
    aspect_ratio = body.get("aspect_ratio", "16:9") 

    results = []
    current_index = start_index

    for i, prompt in enumerate(prompts):
        try:
            images = gemini_image_generation(prompt, count=1,aspect_ratio=aspect_ratio)  # 固定 count=1
            if not images:
                raise ValueError("無圖片返回")

            # ✅ 僅取第一張
            first_img = images[0]
            rel_url = await save_image_to_disk(first_img, current_index)
            results.append({
                "prompt_index": i,
                "prompt": prompt,
                "uploaded_urls": [rel_url],
                "errors": [],
            })
            current_index += 1

        except Exception as e:
            results.append({
                "prompt_index": i,
                "prompt": prompt,
                "uploaded_urls": [],
                "errors": [str(e)],
            })

    ok = sum(1 for r in results if r["uploaded_urls"])
    fail = len(results) - ok
    return {"message": f"{ok} success, {fail} failed", "results": results}
#####
#####
#####
#####
@app.post("/extract_image_prompts", response_model=ExtractOut)
async def extract_image_prompts(payload: ExtractIn):
    # 1️⃣ 依照場景數抽取 image_prompt_X

    prompts = extract_all_image_prompts(script, payload.scene_count)



    print("\n================== [Extract Prompt Debug] ==================")
    print(f"📝 Input Script Length: {len(script)} chars")
    print(f"🔍 Found {len(prompts)} image prompts")

    if not prompts or len(prompts) < payload.scene_count:
        raise HTTPException(
            status_code=422,
            detail="找不到完整的 image_prompts"
        )

    forward = {
        "prompts": prompts,
        "images_per_prompt": payload.images_per_prompt,
        "start_index": payload.start_index,
        "naming": payload.naming,
    }
    # ✅ 在這裡檢查 forward_body 是否可用於 generate_images_from_prompts
    validate_forward_body(forward)
    
    return ExtractOut(
        prompts=prompts,
        images_per_prompt=payload.images_per_prompt,
        start_index=payload.start_index,
        naming=payload.naming,
        forward_body=forward,
    )

@app.post("/extract_then_generate")
async def extract_then_generate(payload: ScriptPayload):
    # 1️⃣ 從腳本文字中抽取 image_prompts
    text = (payload.result or "").strip()
    prompts = parse_image_prompts(text)
    # ★★★ 新增 Log：印出提取結果 ★★★
    print(f"\n{'='*20} [Extract Prompt Debug] {'='*20}")
    print(f"📝 Input Script Length: {len(text)} chars")
    print(f"🔍 Found {len(prompts)} prompts:")
    if not prompts:
        raise HTTPException(status_code=422, detail="找不到 image_prompt。")

    # 2️⃣ 組 forward_body 並立即驗證
    forward_body = {
        "prompts": prompts,
        "images_per_prompt": 1,  # 🔒 固定只生一張
        "start_index": payload.start_index,
        "naming": payload.naming,
        "aspect_ratio": payload.aspect_ratio

    }
    validate_forward_body(forward_body)  # ✅ ← 在這裡被呼叫！

    # 3️⃣ 呼叫實際生圖邏輯（直接呼叫函式，不再發 HTTP）
    generate_result = await generate_images_from_prompts_internal(forward_body)

    # 4️⃣ 整理回傳結果
    uploaded_urls_flat = []
    for item in generate_result["results"]:
        uploaded_urls_flat += item.get("uploaded_urls", [])

    return {
        "forward_body": forward_body,
        "generate_result": generate_result,
        "uploaded_urls_flat": uploaded_urls_flat,
        "n_prompts": len(prompts),
        "images_per_prompt": 1,
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
