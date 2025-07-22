# 图像分类模型训练平台（MVP）

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 启动后端

```bash
uvicorn main:app --reload
```

## 3. 上传数据集接口

- 接口：`POST /upload_datasets/`
- 参数：
  - files: 多个图片压缩包（zip/tar），每个对应一个分类
  - class_names: 多个分类名（与文件一一对应）
- 示例：
  - files: [cat.zip, dog.zip]
  - class_names: [cat, dog]

可以用Postman或curl测试。 