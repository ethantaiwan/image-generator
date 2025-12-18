import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { RefreshCw, Sparkles, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

// API 定義
const API_BASE_URL = "https://image-generator-i03j.onrender.com"; 
const API_IMAGE_GENERATE_STORE = `${API_BASE_URL}/extract_then_generate`; 
const API_IMAGE_EDIT_STORE = `${API_BASE_URL}/edit_image_store`;

// --- 自動縮放的文字框 (保持不變) ---
const AutoResizeTextarea = ({ value, onChange, disabled, placeholder }: any) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight + 2}px`;
    }
  }, [value]);

  return (
    <Textarea
      ref={textareaRef}
      value={value}
      onChange={onChange}
      disabled={disabled}
      placeholder={placeholder}
      rows={1}
      className="min-h-[80px] resize-y overflow-hidden text-sm leading-relaxed"
    />
  );
};

// --- 介面定義 ---
interface ImageGenerationStepProps {
  formData: {
    sceneCount: number;
    aspectRatio: string;
    [key: string]: any;
  };
  scriptResult: string;
  onPrev: () => void;
  onNext: () => void;
}

interface ImageState {
    url: string; 
    prompt: string;
    publicUrl: string;
    error?: string;
}

const ImageGenerationStep = ({ formData, scriptResult, onPrev, onNext }: ImageGenerationStepProps) => {
  const { toast } = useToast();
  
  // 狀態管理
  const [images, setImages] = useState<ImageState[]>([]);
  const [isGenerating, setIsGenerating] = useState(false); // 全局生成狀態
  const [isRegenerating, setIsRegenerating] = useState<number | null>(null); // 單張重繪狀態 (存 index)
  const [editPrompts, setEditPrompts] = useState<string[]>([]);

  // 樣式輔助函數
  const getGridClass = (count: number) => {
    if (count === 1) return "grid-cols-1 max-w-md mx-auto";
    if (count === 2) return "grid-cols-1 md:grid-cols-2";
    if (count <= 4) return "grid-cols-1 md:grid-cols-2"; 
    return "grid-cols-1 sm:grid-cols-2 md:grid-cols-3";
  };

  const getAspectRatioClass = (ratio: string) => {
    switch (ratio) {
        case "9:16": return "aspect-[9/16]";
        case "16:9": return "aspect-video";
        case "1:1":  return "aspect-square";
        case "3:4":  return "aspect-[3/4]";
        case "4:3":  return "aspect-[4/3]";
        default:     return "aspect-video";
    }
  };

  // --- 1. 批量生成圖片 (保持不變) ---
  const generateImages = useCallback(async () => {
    setIsGenerating(true);
    const targetCount = formData.sceneCount || 4;
    
    const payload = {
        result: scriptResult,
        images_per_prompt: 1,
        start_index: 0, 
        naming: "scene",
        aspect_ratio: formData.aspectRatio || "16:9"
    };

    toast({ title: "開始生成", description: `AI 正在繪製 ${targetCount} 張圖 (${payload.aspect_ratio})...` });

    try {
        const response = await fetch(API_IMAGE_GENERATE_STORE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        const data = await response.json();

        // 處理回傳
        const newImages: ImageState[] = data.generate_result.results.flatMap((result: any) => {
            if (!result.uploaded_urls || result.uploaded_urls.length === 0) {
                return [{
                    url: "", 
                    prompt: result.prompt, 
                    publicUrl: "",
                    error: result.errors?.join(", ") || "生成失敗"
                }];
            }
            return result.uploaded_urls.map((relativePath: string) => ({
                url: `${API_BASE_URL}${relativePath}?v=${Date.now()}`,
                prompt: result.prompt, 
                publicUrl: `${API_BASE_URL}${relativePath}`, 
            }));
        });

        // 截斷或補齊
        let finalImages = newImages;
        if (finalImages.length > targetCount) finalImages = finalImages.slice(0, targetCount);
        
        setImages(finalImages);
        setEditPrompts(finalImages.map(img => img.prompt));
        toast({ title: "完成", description: "圖片生成完畢" });

    } catch (error: any) {
        console.error(error);
        toast({ variant: 'destructive', title: "錯誤", description: error.message });
    } finally {
        setIsGenerating(false);
    }
  }, [scriptResult, toast, formData]);

  // --- 2. [核心修復] 單張重繪圖片 (依照您提供的正確版本) ---
  const regenerateImage = useCallback(async (index: number) => {
    
    // --- 設置載入狀態 ---
    setIsRegenerating(index); // 鎖定按鈕

    const currentPrompt = editPrompts[index];
    const targetIndex = index; 
    const currentImage = images[index]; 
    
    toast({
        title: "重新生成照片",
        description: `正在抓取原始圖片並使用新提示詞進行編輯...`,
    });
    
    try {
        // 1. 先抓取目前圖片的 Blob
        const imageResponse = await fetch(currentImage.url);
        if (!imageResponse.ok) {
            throw new Error(`無法抓取原始圖片進行編輯: ${imageResponse.status}`);
        }
        const imageBlob = await imageResponse.blob();

        // 2. 建立 FormData
        // 注意：這裡將變數命名為 submitData 以避免與 props 中的 formData 衝突
        const submitData = new FormData();
        submitData.append('edit_prompt', currentPrompt);
        // 假設後端需要 'file' 這個欄位名
        submitData.append('file', imageBlob, `original_image_${index}.png`); 

        // 3. 發送 API 請求
        const response = await fetch(`${API_IMAGE_EDIT_STORE}?target_index=${targetIndex}`, {
            method: 'POST',
            // fetch 會自動設置 multipart/form-data 的 content-type 和 boundary，所以不需要手動設置 headers
            body: submitData, 
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `API 錯誤: ${response.status}`);
        }

        const data = await response.json();
        
        // 4. 處理回傳結果
        // 假設回傳格式是 { uploaded_urls: ["/static/xxx.png"] }
        const newPublicRelativeUrl = data.uploaded_urls[0]; 
        const absoluteUrl = `${API_BASE_URL}${newPublicRelativeUrl}`;
        
        // 5. 更新 State
        setImages((prevImages) => {
            const newImages = [...prevImages];
            newImages[index] = {
                ...newImages[index],
                url: `${absoluteUrl}?v=${Date.now()}`, // 加入時間戳記防止快取
                publicUrl: absoluteUrl,
                prompt: currentPrompt, // 重要：更新 state 中的 prompt，確保下次重繪用的是最新的
                error: undefined,
            };
            return newImages;
        });
        
        toast({
            title: "照片已更新",
            description: `第 ${index + 1} 張照片已重新生成並儲存。`,
        });

    } catch (error: any) {
        console.error("重新生成失敗:", error);
        toast({
            variant: 'destructive',
            title: "照片編輯失敗",
            description: error.message || "無法連接到伺服器或處理圖片。",
        });
    } finally {
        // --- 解除載入狀態 ---
        setIsRegenerating(null); // 解鎖按鈕
    }
  }, [images, editPrompts, toast]); 

  const updatePrompt = (index: number, value: string) => {
    const newPrompts = [...editPrompts];
    newPrompts[index] = value;
    setEditPrompts(newPrompts);
  };

  return (
    <Card className="max-w-6xl mx-auto bg-accent/10 border-primary/20">
      <CardContent className="p-8">
        <h2 className="text-2xl font-semibold mb-2 text-center">AI 照片生成</h2>
        
        {images.length === 0 ? (
          <div className="text-center py-12">
            <Button onClick={generateImages} disabled={isGenerating} className="px-8 py-6 text-lg">
              <Sparkles className={`w-5 h-5 mr-2 ${isGenerating ? 'animate-spin' : ''}`} />
              {isGenerating ? "生成中..." : "開始生成照片"}
            </Button>
          </div>
        ) : (
          <div className="space-y-8">
            <div className={`grid gap-6 ${getGridClass(images.length)}`}>
              {images.map((image, index) => (
                <Card key={index} className="bg-card border-primary/20 flex flex-col">
                  <CardContent className="p-4 flex-1 flex flex-col">
                    
                    {/* 圖片顯示區 */}
                    <div className={`w-full bg-muted rounded-lg overflow-hidden mb-4 relative group ${getAspectRatioClass(formData.aspectRatio)}`}>
                      {/* 如果正在全體生成 OR 正在重繪這一張，顯示 Loading */}
                      {(isGenerating || isRegenerating === index) ? (
                        <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground bg-black/5">
                            <Sparkles className="w-8 h-8 mb-2 animate-spin text-primary" />
                            <span>{isRegenerating === index ? "重繪中..." : "生成中..."}</span>
                        </div>
                      ) : image.url ? (
                        <img src={image.url} alt={`Scene ${index + 1}`} className="w-full h-full object-cover transition-all duration-300 hover:scale-105" />
                      ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center bg-red-50 text-red-500 p-2 text-center">
                            <AlertCircle className="w-8 h-8 mb-2" />
                            <span className="text-xs">{image.error}</span>
                        </div>
                      )}
                      <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded z-10">Scene {index + 1}</div>
                    </div>
                    
                    <div className="space-y-3 mt-auto">
                      {/* 自動縮放 Textarea (保持不變) */}
                      <AutoResizeTextarea
                        value={editPrompts[index]} 
                        onChange={(e: any) => updatePrompt(index, e.target.value)} 
                        disabled={isGenerating || isRegenerating === index}
                        placeholder="輸入圖像提示詞..."
                      />

                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => regenerateImage(index)} 
                        className="w-full" 
                        disabled={isGenerating || isRegenerating !== null} // 防止同時重繪多張
                      >
                         <RefreshCw className={`w-4 h-4 mr-2 ${isRegenerating === index ? 'animate-spin' : ''}`} /> 
                         {isRegenerating === index ? "正在重繪..." : "重新生成"}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="flex justify-center space-x-4 pt-4">
                <Button variant="outline" onClick={onPrev} disabled={isGenerating || isRegenerating !== null}>← 上一步</Button>
                <Button variant="outline" onClick={generateImages} disabled={isGenerating || isRegenerating !== null}>
                    <RefreshCw className="w-4 h-4 mr-2"/>全部重繪
                </Button>
                <Button onClick={onNext} disabled={isGenerating || isRegenerating !== null}>下一步 →</Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export { ImageGenerationStep };
