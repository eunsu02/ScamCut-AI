import cv2
import yt_dlp
import os
from pathlib import Path

# 얼굴 탐지기 초기화
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def process_youtube_video(url: str):
    # 1. 유튜브 영상 다운로드 (최저 화질로 빠르게)
    ydl_opts = {"format": "worst", "outtmpl": "temp_video.mp4", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # 2. 영상에서 프레임 추출 (중간 프레임 1개만 예시로)
    cap = cv2.VideoCapture("temp_video.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
    ret, frame = cap.read()
    cap.release()
    os.remove("temp_video.mp4")  # 임시 파일 삭제

    if not ret:
        return None

    # 3. 얼굴 영역 Crop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    return frame[y : y + h, x : x + w]
