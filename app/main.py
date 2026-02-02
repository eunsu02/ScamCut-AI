from fastapi import FastAPI, File, UploadFile
from app.model_loader import predict_deepfake
from fastapi import FastAPI, HTTPException
from app.model_loader import get_model
from app.utils import process_youtube_video
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np

app = FastAPI(title="ScamGuard AI API")

model, device = get_model("models/scamguard_model.pth")


@app.get("/")
def read_root():
    return {"message": "Scam Guard AI Server is Running!"}


transformer = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


@app.post("/deepfake")
async def predict_deepfake_from_url(url: str):
    # 1. 유튜브에서 얼굴 추출
    face_img = process_youtube_video(url)
    if face_img is None:
        raise HTTPException(
            status_code=400, detail="얼굴을 찾을 수 없거나 영상 처리에 실패했습니다."
        )

    # 2. 전처리 및 추론
    input_tensor = transformer(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()

    return {
        "url": url,
        "is_fake": prob > 0.5,
        "confidence": round(prob * 100, 2),
        "message": "🚨 딥페이크 의심" if prob > 0.5 else "✅ 정상 영상",
    }


@app.get("/test-batch")
async def test_batch_images():
    # 1. 테스트 이미지 폴더 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "test_images")

    if not os.path.exists(test_dir):
        return {"error": "test_images 폴더를 찾을 수 없습니다."}

    results = []
    # 2. 폴더 내 파일들 리스팅 (png, jpg, jpeg만 골라내기)
    image_files = [
        f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in image_files:
        img_path = os.path.join(test_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # 💡 PIL 이미지를 Numpy 배열로 변환해서 transformer에 전달
        image_np = np.array(image)
        input_tensor = transformer(image_np).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()

        confidence = round(prob * 100, 2)
        results.append(
            {
                "filename": filename,
                "is_fake": confidence > 50,
                "confidence": f"{confidence}%",
                "status": "🚨 딥페이크 의심" if confidence > 50 else "✅ 정상",
            }
        )

    # 4. 전체 결과 반환
    return {"total_count": len(results), "predictions": results}
