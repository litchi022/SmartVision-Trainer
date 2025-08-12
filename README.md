# 智能图像分类训练平台（MVP）

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

> 🚀 一个开箱即用的智能图像分类训练与实时预测平台

基于 **FastAPI + PyTorch** 构建的全栈 AI 解决方案，提供前端可视化数据上传、模型训练进度监控、摄像头实时推理预测，以及项目一键导入/导出等完整功能。支持迁移学习快速训练自定义分类器，无需编写代码即可完成从数据到部署的全流程。

## 📖 目录

- [功能特性](#-功能特性)
- [技术栈](#-技术栈)
- [演示截图](#-演示截图)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [详细安装](#-详细安装)
- [前端操作流程](#-前端操作流程)
- [API 接口文档](#-api-接口文档)
- [项目结构](#-项目结构)
- [训练与模型](#-训练与模型)
- [部署指南](#-部署指南)
- [性能优化](#-性能优化)
- [常见问题](#-常见问题)
- [开发指南](#-开发指南)
- [贡献指南](#-贡献指南)
- [更新日志](#-更新日志)
- [许可证](#-许可证)

## ✨ 功能特性

### 🎯 核心功能
- **🖼️ 可视化数据上传**：支持多分类批量图片上传，实时进度显示与文件预览
- **🤖 智能模型训练**：基于 ResNet18 的迁移学习，异步后台训练与实时进度监控  
- **📹 实时视频预测**：浏览器摄像头实时推理，多分类置信度可视化展示
- **📦 项目全生命周期管理**：一键导出/导入项目（数据集+模型），便于分享与备份

### 🔧 技术特性
- **异步处理**：FastAPI 异步框架，支持高并发请求处理
- **WebSocket 实时通信**：训练进度与预测结果实时推送
- **迁移学习**：预训练模型快速收敛，小样本数据集也能获得良好效果
- **跨平台支持**：Windows/Linux/macOS 全平台兼容
- **零配置启动**：开箱即用，无需复杂环境配置

## 🛠️ 技术栈

### 后端技术
- **Web 框架**: FastAPI 0.100+
- **深度学习**: PyTorch 2.0+ / TorchVision
- **异步处理**: Python asyncio
- **文件处理**: Pillow (PIL)
- **数据序列化**: JSON

### 前端技术  
- **界面**: 原生 HTML5 + CSS3
- **交互**: Vanilla JavaScript (ES6+)
- **实时通信**: WebSocket API
- **媒体处理**: Canvas API / Media Devices API

### 开发工具
- **代码规范**: Python PEP 8
- **依赖管理**: pip + requirements.txt
- **开发服务器**: Uvicorn (ASGI)

## 📸 演示截图

> 💡 **提示**: 以下为界面功能演示，实际效果以运行结果为准

### 主界面总览
```
┌─────────────────────────────────────────────────────┐
│                  图像分类器                           │
├─────────────────────────────────────────────────────┤
│  第一步：上传数据集                                    │  
│  ┌─ 分类名称: [cat        ] ──────────────────────┐   │
│  │  图片文件: [选择文件...] (已选择 15 个文件)      │   │
│  └─ 分类名称: [dog        ] ──────────────────────┘   │
│      图片文件: [选择文件...] (已选择 12 个文件)        │
│  [添加更多分类] [上传数据集]                           │
├─────────────────────────────────────────────────────┤
│  第二步：训练模型                                      │
│  [开始训练] ████████░░ 80% (Epoch 8/10)              │
├─────────────────────────────────────────────────────┤  
│  第三步：实时预测                                      │
│  [启动摄像头] [关闭摄像头]                             │
│  ┌─────────────┐ 预测: cat (95%)                     │
│  │   📹 Video   │ cat  ████████████ 95%              │
│  │    Frame     │ dog  ██░░░░░░░░░░ 5%               │
│  └─────────────┘                                     │
└─────────────────────────────────────────────────────┘
```

### 训练进度监控
- 实时 Epoch 进度条与损失曲线
- 验证集准确率实时更新  
- 详细训练日志滚动显示

### 实时预测界面
- 摄像头视频流实时显示
- 多分类置信度条形图
- 最高置信度分类结果高亮

## 💻 系统要求

### 基础环境
- **操作系统**: Windows 10+ / Ubuntu 18.04+ / macOS 10.15+
- **Python**: 3.9+ (推荐 3.10 或 3.11)
- **内存**: 最低 4GB，推荐 8GB 以上
- **存储空间**: 至少 2GB 可用空间
- **网络**: 首次运行需联网下载预训练模型

### 硬件加速 (可选)
- **CPU**: 支持 AVX 指令集的现代处理器
- **GPU**: NVIDIA GPU + CUDA 11.8+ (可选，显著提升训练速度)
- **摄像头**: 支持 WebRTC 的摄像头设备 (用于实时预测)

### 浏览器支持
- **推荐**: Chrome 90+ / Edge 90+ / Firefox 88+
- **功能要求**: 支持 WebSocket + MediaDevices API

## 🚀 快速开始

### 🔥 一键启动 (推荐)

```bash
# 1️⃣ 克隆项目
git clone https://github.com/your-username/image-classification-platform.git
cd image-classification-platform

# 2️⃣ 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS  
source venv/bin/activate

# 3️⃣ 安装依赖
pip install -r requirements.txt

# 4️⃣ 启动服务
uvicorn main:app --reload

# 5️⃣ 打开浏览器
# 访问 http://127.0.0.1:8000
```

### ⚡ Docker 快速部署 (即将支持)

```bash
# 构建镜像
docker build -t image-classifier .

# 启动容器
docker run -p 8000:8000 image-classifier
```

## 🔧 详细安装

### 步骤 1: 环境准备

#### 使用 Conda (推荐)
```bash
# 创建专用环境
conda create -n image-classifier python=3.10
conda activate image-classifier

# 安装 PyTorch (CPU 版本)
conda install pytorch torchvision cpuonly -c pytorch

# 或安装 GPU 版本 (需要 CUDA)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 使用 pip + venv
```bash
# 创建虚拟环境
python -m venv image-classifier-env

# 激活环境 (Windows)
image-classifier-env\Scripts\activate

# 激活环境 (Linux/macOS)
source image-classifier-env/bin/activate

# 升级 pip
pip install --upgrade pip
```

### 步骤 2: 安装依赖包

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或手动安装
pip install fastapi>=0.100.0
pip install uvicorn[standard]
pip install torch>=2.0.0
pip install torchvision>=0.15.0  
pip install python-multipart
pip install pillow
```

### 步骤 3: 验证安装

```bash
# 检查 Python 环境
python --version  # 应该显示 3.9+

# 检查 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 检查 FastAPI 安装  
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

## 📂 项目结构
```text
图像分类训练平台/
  ├─ 📁 datasets/              # 数据集存储目录
  │   ├─ 📁 class1/           # 分类1的图片文件
  │   ├─ 📁 class2/           # 分类2的图片文件  
  │   └─ 📁 ...               # 更多分类
  ├─ 📁 models/               # 模型存储目录
  │   ├─ 📁 20240101-120000/  # 训练时间戳命名的模型目录
  │   │   ├─ 🔹 model.pth     # PyTorch 模型权重文件
  │   │   └─ 🔹 class_map.json # 类别到索引映射文件
  │   └─ 📁 ...               # 历史训练模型
  ├─ 📁 static/               # 前端静态资源
  │   ├─ 🌐 index.html        # 主界面 HTML
  │   └─ ⚡ script.js         # 前端交互逻辑
  ├─ 🐍 main.py              # FastAPI 主程序入口
  ├─ 🤖 train.py             # 模型训练核心逻辑  
  ├─ 📋 requirements.txt      # Python 依赖清单
  ├─ 📖 README.md            # 项目说明文档
  └─ 📝 智能图像分类训练平台.md # 设计文档 (可选)
```

### 📊 目录说明

| 目录/文件 | 用途 | 说明 |
|-----------|------|------|
| `datasets/` | 训练数据存储 | 按分类自动创建子目录，支持 JPG/PNG 格式 |
| `models/` | 模型产物存储 | 每次训练生成时间戳目录，包含权重和映射文件 |
| `static/` | 前端界面资源 | 单页面应用，支持数据上传、训练监控、实时预测 |
| `main.py` | 后端服务入口 | FastAPI 路由定义、WebSocket 处理、文件管理 |
| `train.py` | 训练引擎 | ResNet18 迁移学习、数据增强、模型保存逻辑 |

## 🎮 前端操作流程

### 📤 步骤1: 数据集上传
1. **添加分类**: 点击"添加更多分类"创建多个分类组
2. **选择图片**: 每个分类选择多张图片文件 (支持批量选择)
3. **预览确认**: 界面显示已选文件列表，可单独移除
4. **开始上传**: 点击"上传数据集"，观察实时进度条

> 💡 **最佳实践**: 每个分类建议上传 20-100 张图片，确保样本均衡分布

### 🏋️ 步骤2: 模型训练  
1. **启动训练**: 上传完成后，"开始训练"按钮将激活
2. **监控进度**: 实时查看训练 Epoch 进度和损失数值
3. **查看日志**: 训练过程详细信息在日志区域滚动显示
4. **等待完成**: 训练完成后自动保存模型到 `models/` 目录

> ⏱️ **时间估算**: 通常 2-3 个分类各 50 张图片的训练需要 5-10 分钟 (CPU)

### 📸 步骤3: 实时预测
1. **授权摄像头**: 点击"启动摄像头"，浏览器请求摄像头权限
2. **开始推理**: 摄像头启动后自动开始实时图像分类
3. **查看结果**: 观察各分类置信度条形图和最高分预测结果
4. **停止预测**: 点击"关闭摄像头"结束实时预测

> 🔒 **隐私说明**: 所有图像处理在本地进行，不会上传到外部服务器

### 📦 步骤4: 项目管理
- **导出项目**: 一键打包 `datasets/` 和 `models/` 为 ZIP 文件
- **导入项目**: 上传之前导出的 ZIP 文件，快速恢复训练环境
- **数据清理**: 支持删除指定分类或清空所有数据集

## 📡 API 接口文档

### 📤 1. 数据集上传

**接口路径**: `POST /upload_datasets/`

**请求参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `files` | File[] | ✅ | 图片文件数组 (支持 JPG/PNG/WEBP) |
| `class_names` | String[] | ✅ | 分类名称数组 (与files一一对应) |

**请求示例**:
```bash
# cURL 示例
curl -X POST http://127.0.0.1:8000/upload_datasets/ \
  -F "files=@cat_1.jpg" -F "class_names=cat" \
  -F "files=@cat_2.jpg" -F "class_names=cat" \
  -F "files=@dog_1.jpg" -F "class_names=dog"

# Python requests 示例  
import requests
files = [('files', open('cat1.jpg', 'rb')), ('files', open('dog1.jpg', 'rb'))]
data = {'class_names': ['cat', 'dog']}
response = requests.post('http://127.0.0.1:8000/upload_datasets/', files=files, data=data)
```

**响应示例**:
```json
{
  "message": "上传成功",
  "files": [
    {"class_name": "cat", "file_path": "datasets/cat/cat_1.jpg"},
    {"class_name": "cat", "file_path": "datasets/cat/cat_2.jpg"},
    {"class_name": "dog", "file_path": "datasets/dog/dog_1.jpg"}
  ]
}
```

**错误响应**:
```json
{
  "error": "文件数量与分类名数量不一致",
  "detail": "files: 3, class_names: 2"
}
```

### 🤖 2. 启动模型训练

**接口路径**: `POST /train/`

**功能说明**: 启动后台异步训练任务，返回会话ID用于进度跟踪

**请求示例**:
```bash
# cURL 示例
curl -X POST http://127.0.0.1:8000/train/

# Python requests 示例
import requests
response = requests.post('http://127.0.0.1:8000/train/')
print(response.json())
```

**响应示例**:
```json
{
  "message": "模型训练已开始。",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**错误响应**:
```json
{
  "detail": "数据集为空，请先上传数据。"
}
```

### 📊 3. 训练进度监控 (WebSocket)

**接口路径**: `WebSocket /ws/train_progress/{session_id}`

**连接示例**:
```javascript
// 前端 JavaScript 示例
const sessionId = "550e8400-e29b-41d4-a716-446655440000";
const socket = new WebSocket(`ws://127.0.0.1:8000/ws/train_progress/${sessionId}`);

socket.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    console.log(`训练进度: ${progress.current_epoch}/${progress.total_epochs}`);
};
```

**实时消息格式**:
```json
{
  "status": "Training",
  "current_epoch": 5,
  "total_epochs": 10, 
  "log": "Epoch 5/10 | Train Loss: 0.2145 | Val Loss: 0.2890 | Val Acc: 0.9250"
}
```

**完成消息格式**:
```json
{
  "status": "Completed",
  "current_epoch": 10,
  "total_epochs": 10,
  "log": "Training complete. Model saved to models/20240101-120000"
}
```

### 🎥 4. 实时视频预测 (WebSocket)

**接口路径**: `WebSocket /ws/predict`

**交互流程**:
1. **建立连接** → 服务器推送可用分类名称
2. **发送图片帧** → 客户端发送二进制图像数据  
3. **接收预测** → 服务器返回各分类置信度

**连接示例**:
```javascript
// 前端 JavaScript 示例
const socket = new WebSocket('ws://127.0.0.1:8000/ws/predict');

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.class_names) {
        // 首次连接，接收分类名称
        console.log('可用分类:', data.class_names);
    } else if (data.confidences) {
        // 预测结果
        console.log('预测结果:', data.confidences);
    }
};

// 发送图像帧 (从 Canvas 获取)
canvas.toBlob(blob => {
    socket.send(blob);
}, 'image/jpeg');
```

**初始化消息**:
```json
{
  "class_names": ["cat", "dog", "bird"]
}
```

**预测结果消息**:
```json
{
  "confidences": {
    "cat": 0.856,
    "dog": 0.132, 
    "bird": 0.012
  }
}
```

### 🖼️ 5. 单图片预测 (REST)

**接口路径**: `POST /predict/`

**请求参数**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `file` | File | ✅ | 单张图片文件 |

**请求示例**:
```bash
# cURL 示例
curl -X POST http://127.0.0.1:8000/predict/ -F "file=@test.jpg"

# Python requests 示例
import requests
with open('test.jpg', 'rb') as f:
    response = requests.post('http://127.0.0.1:8000/predict/', files={'file': f})
    print(response.json())
```

**响应示例**:
```json
{
  "predicted_class": "cat",
  "confidence": 0.9731,
  "model_used": "20240101-120000"
}
```

**错误响应**:
```json
{
  "detail": "没有找到训练好的模型。请先训练一个模型。"
}
```

### 📦 6. 项目管理

#### 6.1 导出项目
**接口路径**: `GET /export_project/`

**功能说明**: 将当前 `datasets/` 和 `models/` 目录打包为 ZIP 文件下载

**请求示例**:
```bash
# 浏览器直接访问下载
http://127.0.0.1:8000/export_project/

# wget 命令下载
wget http://127.0.0.1:8000/export_project/ -O project_backup.zip
```

#### 6.2 导入项目  
**接口路径**: `POST /import_project/`

**功能说明**: 上传 ZIP 文件并覆盖当前项目数据

**请求示例**:
```bash
# cURL 上传
curl -X POST http://127.0.0.1:8000/import_project/ \
     -F "file=@project_export_20240101-120000.zip"
```

**响应示例**:
```json
{
  "message": "项目导入成功！页面将重新加载。"
}
```

### 📊 7. 系统状态与管理

#### 7.1 获取系统状态
**接口路径**: `GET /status`

**响应示例**:
```json
{
  "datasets_available": true,
  "class_names": ["cat", "dog", "bird"],
  "model_available": true
}
```

#### 7.2 删除指定分类
**接口路径**: `DELETE /dataset/{class_name}`

**请求示例**:
```bash
curl -X DELETE http://127.0.0.1:8000/dataset/cat
```

#### 7.3 清空所有数据集
**接口路径**: `DELETE /datasets/all`

**请求示例**:
```bash
curl -X DELETE http://127.0.0.1:8000/datasets/all
```

## 🧠 训练与模型

### 模型架构
- **基础模型**: ResNet18 (ImageNet 预训练)
- **迁移学习**: 替换最后全连接层适配自定义分类数
- **输入尺寸**: 224×224 RGB 图像
- **预处理**: Resize + ToTensor + ImageNet 标准化

### 训练配置
```python
# 默认训练参数
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
OPTIMIZER = "Adam"
LOSS_FUNCTION = "CrossEntropyLoss"
```

### 模型文件结构
```text
models/20240101-120000/
  ├─ model.pth          # PyTorch 模型权重 (state_dict)
  ├─ class_map.json     # 类别名到索引映射
  └─ training_log.txt   # 训练日志 (可选)
```

### 性能指标
| 数据集规模 | 训练时间 (CPU) | 训练时间 (GPU) | 预期准确率 |
|-----------|---------------|---------------|-----------|
| 2 类 × 50 图片 | ~5 分钟 | ~1 分钟 | 85-95% |
| 5 类 × 100 图片 | ~15 分钟 | ~3 分钟 | 80-90% |
| 10 类 × 200 图片 | ~45 分钟 | ~8 分钟 | 75-85% |

## 🚀 部署指南

### 🐳 Docker 部署 (推荐生产环境)

1. **创建 Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **构建并运行**:
```bash
docker build -t image-classifier .
docker run -p 8000:8000 -v ./data:/app/datasets -v ./models:/app/models image-classifier
```

### ☁️ 云平台部署

#### Heroku 部署
```bash
# 安装 Heroku CLI 后
heroku create your-app-name
git push heroku main
heroku ps:scale web=1
```

#### Railway 部署  
1. 连接 GitHub 仓库到 Railway
2. 设置环境变量 `PORT=8000`
3. 自动部署完成

### 🔧 Nginx 反向代理配置
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## ⚡ 性能优化

### 🖥️ 服务器端优化
- **异步处理**: 使用 FastAPI 异步特性处理并发请求
- **模型缓存**: 模型加载后保持在内存中，避免重复加载
- **批量推理**: WebSocket 连接复用，减少模型初始化开销
- **静态文件**: 使用 CDN 或 Nginx 服务静态资源

### 🧠 模型优化
- **模型量化**: 使用 PyTorch 量化减少模型大小
- **模型剪枝**: 移除不重要的连接降低计算量
- **ONNX 转换**: 转换为 ONNX 格式提升推理速度
- **TensorRT**: NVIDIA GPU 环境下使用 TensorRT 加速

### 📱 前端优化
- **图像压缩**: Canvas 压缩图像后再发送
- **连接池**: WebSocket 连接复用
- **缓存策略**: 利用浏览器缓存静态资源
- **懒加载**: 大文件按需加载

## ❓ 常见问题 (FAQ)

### 🔧 安装问题
**Q: 无法安装 PyTorch/TorchVision？**  
A: 根据系统环境选择合适版本：
- **CPU 版本**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- **CUDA 版本**: 访问 [PyTorch 官网](https://pytorch.org) 获取匹配命令

**Q: 提示 "Microsoft Visual C++ 14.0 is required" ？**  
A: Windows 用户需要安装 Visual Studio Build Tools 或使用预编译的 wheel 包

### 📹 摄像头问题
**Q: 摄像头权限被拒绝？**  
A: 检查浏览器设置允许摄像头访问，推荐使用 HTTPS 或 localhost

**Q: 摄像头画面黑屏？**  
A: 确认摄像头未被其他应用占用，尝试刷新页面重新授权

### 🤖 模型问题  
**Q: 训练后准确率很低？**  
A: 检查数据质量、增加训练样本数量、确保类别均衡分布

**Q: 预测结果不准确？**  
A: 确认测试图片与训练数据分布一致，考虑增加数据增强

### 🌐 网络问题
**Q: 端口 8000 被占用？**  
A: 使用 `uvicorn main:app --port 9000` 更换端口

**Q: WebSocket 连接失败？**  
A: 检查防火墙设置，确认 WebSocket 请求未被代理拦截

## 🛠️ 开发指南

### 项目贡献
1. Fork 项目到个人仓库
2. 创建功能分支 `git checkout -b feature/amazing-feature`
3. 提交更改 `git commit -m 'Add amazing feature'`
4. 推送分支 `git push origin feature/amazing-feature`  
5. 创建 Pull Request

### 代码规范
- 遵循 PEP 8 Python 代码规范
- 使用 Black 自动格式化代码
- 编写单元测试确保功能正常
- 更新文档说明新增功能

### 本地开发  
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 代码格式化
black *.py

# 运行测试
pytest tests/

# 启动开发服务器
uvicorn main:app --reload --log-level debug
```

## 📚 贡献指南

### 欢迎贡献
我们欢迎社区贡献，包括但不限于：
- 🐛 Bug 修复
- ✨ 新功能开发  
- 📖 文档完善
- 🎨 UI/UX 改进
- 🧪 测试用例

### 提交规范
请遵循 [Conventional Commits](https://conventionalcommits.org/) 规范：
- `feat:` 新功能
- `fix:` Bug 修复  
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具链更新

## 📋 更新日志

### v1.0.0 (2024-01-01)
- ✨ 初始版本发布
- 🖼️ 支持多分类图像上传
- 🤖 ResNet18 迁移学习训练
- 📹 实时摄像头预测  
- 📦 项目导入导出功能
- 🌐 Web 界面完整实现

### 规划中的功能 (Roadmap)
- 🔄 支持更多预训练模型 (EfficientNet, Vision Transformer)
- 📊 训练过程可视化图表
- 🎯 数据增强配置界面
- 📱 移动端适配
- 🔍 模型解释性分析
- 📂 批量图片预测接口

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

```text
MIT License

Copyright (c) 2024 图像分类训练平台

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**如果这个项目对您有帮助，请给一个 ⭐ Star！**

[📋 报告问题](../../issues) · [💡 功能建议](../../discussions) · [📖 查看文档](../../wiki)

Made with ❤️ by [Your Name](https://github.com/your-username)

</div>
