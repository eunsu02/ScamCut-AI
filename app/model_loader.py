import torch
import io
from PIL import Image


# 나중에 코랩에서 학습한 모델을 여기서 로드할 예정
def predict_deepfake(image_bytes):
    # 테스트용 응답
    return {"is_scam": True, "probability": 0.88}
