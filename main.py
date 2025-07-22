from fastapi import (
    FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException,
    WebSocket, WebSocketDisconnect
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import shutil
import json
from io import BytesIO
import io
import zipfile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import asyncio
from uuid import uuid4
import time

from train import train_model

app = FastAPI()

# In-memory storage for training progress
training_progress = {}

# Mount the static directory to serve index.html, script.js, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

DATASET_DIR = "datasets"
MODELS_DIR = "models"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Helper function to get the latest model path
def get_latest_model_dir():
    if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
        return None
    latest_model_dir = max(
        [os.path.join(MODELS_DIR, d) for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))],
        key=os.path.getmtime,
    )
    return latest_model_dir

# Helper function to transform image
def transform_image(image_bytes: bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.post("/upload_datasets/")
async def upload_datasets(
    files: List[UploadFile] = File(...),
    class_names: List[str] = Form(...)
):
    if len(files) != len(class_names):
        return JSONResponse(status_code=400, content={"error": "文件数量与分类名数量不一致"})
    saved_files = []
    for file, class_name in zip(files, class_names):
        save_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append({"class_name": class_name, "file_path": file_path})
    return {"message": "上传成功", "files": saved_files}

@app.post("/train/")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers a new model training session in the background.
    """
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        raise HTTPException(status_code=400, detail="数据集为空，请先上传数据。")
    
    session_id = str(uuid4())
    training_progress[session_id] = {
        "status": "Starting",
        "current_epoch": 0,
        "total_epochs": 0,
        "log": "Initializing training..."
    }
    
    background_tasks.add_task(
        train_model, 
        progress=training_progress[session_id],
        data_dir=DATASET_DIR, 
        models_dir=MODELS_DIR
    )
    return {"message": "模型训练已开始。", "session_id": session_id}

@app.websocket("/ws/train_progress/{session_id}")
async def websocket_training_progress(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in training_progress:
        await websocket.send_json({"error": "Invalid session ID"})
        await websocket.close()
        return

    try:
        while True:
            progress = training_progress.get(session_id, {})
            await websocket.send_json(progress)
            if progress.get("status") == "Completed":
                # Clean up after reporting completion
                del training_progress[session_id]
                break
            await asyncio.sleep(1) # Send update every second
    except WebSocketDisconnect:
        print(f"Training progress client for session {session_id} disconnected")
    except Exception as e:
        print(f"Error in training progress websocket: {e}")
        await websocket.close(code=1011)

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    
    latest_model_dir = get_latest_model_dir()
    if not latest_model_dir:
        await websocket.send_json({"error": "没有找到训练好的模型。"})
        await websocket.close()
        return

    model_path = os.path.join(latest_model_dir, "model.pth")
    class_map_path = os.path.join(latest_model_dir, "class_map.json")

    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        await websocket.send_json({"error": "模型文件或分类映射文件不存在。"})
        await websocket.close()
        return
        
    try:
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        # First, send the class names to the client to build the UI
        await websocket.send_json({"class_names": list(idx_to_class.values())})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        while True:
            image_bytes = await websocket.receive_bytes()
            tensor = transform_image(image_bytes).to(device)
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Create a dictionary of class confidences
                confidences = {
                    idx_to_class[i]: prob.item() for i, prob in enumerate(probabilities)
                }
                
                await websocket.send_json({"confidences": confidences})

    except WebSocketDisconnect:
        print(f"Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason=str(e))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of an image using the latest trained model.
    """
    latest_model_dir = get_latest_model_dir()
    if not latest_model_dir:
        raise HTTPException(status_code=404, detail="没有找到训练好的模型。请先训练一个模型。")

    model_path = os.path.join(latest_model_dir, "model.pth")
    class_map_path = os.path.join(latest_model_dir, "class_map.json")

    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        raise HTTPException(status_code=404, detail="模型文件或分类映射文件不存在。")

    # Load class mapping
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Load model architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Read and transform image
    image_bytes = await file.read()
    tensor = transform_image(image_bytes).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = idx_to_class[predicted_idx.item()]
        
    return {
        "predicted_class": predicted_class,
        "confidence": confidence.item(),
        "model_used": os.path.basename(latest_model_dir)
    }

@app.get("/export_project/")
async def export_project():
    """Creates a zip archive of the datasets and models directories and returns it."""
    zip_io = io.BytesIO()
    
    # These are the directories we want to include in the export
    dirs_to_zip = [DATASET_DIR, MODELS_DIR]

    with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as temp_zip:
        for dir_name in dirs_to_zip:
            if not os.path.isdir(dir_name):
                continue
            for root, _, files in os.walk(dir_name):
                for file in files:
                    file_path = os.path.join(root, file)
                    # The arcname is the path inside the zip file
                    arcname = os.path.relpath(file_path, start='.')
                    temp_zip.write(file_path, arcname)

    # Rewind the buffer to the beginning
    zip_io.seek(0)
    
    return StreamingResponse(
        zip_io,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=project_export_{time.strftime('%Y%m%d-%H%M%S')}.zip"}
    )

@app.post("/import_project/")
async def import_project(file: UploadFile = File(...)):
    """Imports a project from a .zip file, overwriting existing data."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="无效的文件格式，请上传 .zip 文件。")

    # Clear existing data
    for dir_to_clear in [DATASET_DIR, MODELS_DIR]:
        if os.path.isdir(dir_to_clear):
            shutil.rmtree(dir_to_clear)
        os.makedirs(dir_to_clear, exist_ok=True)
    
    try:
        # Read the uploaded file into a BytesIO object
        zip_content = await file.read()
        zip_io = io.BytesIO(zip_content)
        
        with zipfile.ZipFile(zip_io, 'r') as temp_zip:
            # Basic validation: check if expected directories are in the zip
            if not any(name.startswith(DATASET_DIR) or name.startswith(MODELS_DIR) for name in temp_zip.namelist()):
                 raise HTTPException(status_code=400, detail="压缩文件内容不符合项目结构。")
            temp_zip.extractall('.')
            
        return {"message": "项目导入成功！页面将重新加载。"}

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="上传的不是一个有效的 .zip 文件。")
    except Exception as e:
        # Restore clean directories in case of failure
        for dir_to_clear in [DATASET_DIR, MODELS_DIR]:
            if os.path.isdir(dir_to_clear):
                shutil.rmtree(dir_to_clear)
            os.makedirs(dir_to_clear, exist_ok=True)
        raise HTTPException(status_code=500, detail=f"导入过程中发生错误: {e}")

@app.get("/status")
async def get_status():
    """Returns the current status of the server, like dataset and model availability."""
    datasets_available = os.path.isdir(DATASET_DIR) and bool(os.listdir(DATASET_DIR))
    
    class_names = []
    if datasets_available:
        class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    model_available = get_latest_model_dir() is not None
    return {
        "datasets_available": datasets_available,
        "class_names": class_names,
        "model_available": model_available
    }

@app.delete("/dataset/{class_name}")
async def delete_dataset_class(class_name: str):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        raise HTTPException(status_code=404, detail="指定的分类不存在。")
    
    try:
        shutil.rmtree(class_dir)
        return {"message": f"分类 '{class_name}' 已被成功删除。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除分类时出错: {e}")

@app.delete("/datasets/all")
async def delete_all_datasets():
    try:
        if os.path.isdir(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
        os.makedirs(DATASET_DIR, exist_ok=True)
        return {"message": "所有数据集已被成功清空。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空数据集时出错: {e}") 