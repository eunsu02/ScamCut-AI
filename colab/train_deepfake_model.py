import os
import cv2
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# 1. 환경 설정 및 경로 정의
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_DIR = Path("/content/drive/MyDrive/FaceForensics")
TRAIN_IMG_DIR = WORK_DIR / "train_images"
TEST_IMG_DIR = WORK_DIR / "test_images"
MODEL_SAVE_PATH = WORK_DIR / "scamguard_model.pth"

# 얼굴 탐지기 (OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 전처리 설정 (Xception 규격)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==========================================
# 2. 데이터 전처리 (Face Extraction)
# ==========================================
def extract_faces(video_path, label, save_base_dir, num_frames=20):
    video_id = video_path.stem
    save_dir = save_base_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for idx in indices:
        save_file_path = save_dir / f"{video_id}_frame{idx}.jpg"
        if save_file_path.exists(): continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(str(save_file_path), face_img)
            break
    cap.release()

# ==========================================
# 3. 모델 학습 엔진 (Train)
# ==========================================
def train_model():
    dataset = datasets.ImageFolder(str(TRAIN_IMG_DIR), transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = timm.create_model('xception', pretrained=True, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# ==========================================
# 4. 검증 및 정확도 측정 (Evaluation)
# ==========================================
def evaluate_model():
    model = timm.create_model('xception', num_classes=2)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    test_dataset = datasets.ImageFolder(str(TEST_IMG_DIR), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n[Final Evaluation Result]\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    # 데이터셋 구축부터 학습, 평가까지의 워크플로우를 실행합니다.
    # train_model()
    # evaluate_model()
    pass

