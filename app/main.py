from fastapi import FastAPI, File, UploadFile
from app.model_loader import predict_deepfake

app = FastAPI(title="ScamGuard AI API")


@app.get("/")
def read_root():
    return {"message": "Scam Guard AI Server is Running!"}


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()

    result = predict_deepfake(contents)
    return {
        "filename": file.filename,
        "is_scam": result["is_scam"],
        "confidence": result["probability"],
    }
