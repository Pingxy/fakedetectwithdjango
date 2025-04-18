import os
import json
import time
import logging
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import torch
import torchvision.transforms as transforms

import sys
sys.path.append("/root/UniversalFakeDetect")
from models import get_model

# 设置日志
logger = logging.getLogger(__name__)

# 模型字典
MODELS = {
    "Imagenet:vit_b_16": "/root/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
}

# 全局变量缓存已加载的模型
LOADED_MODELS = {}

def get_or_load_model(model_name):
    """获取已加载的模型或加载新模型"""
    if model_name not in LOADED_MODELS:
        logger.info(f"Loading model: {model_name}")
        model = get_model(model_name)
        state_dict = torch.load(MODELS[model_name], map_location='cpu')
        model.fc.load_state_dict(state_dict)
        model.eval()
        LOADED_MODELS[model_name] = model
        logger.info(f"Model {model_name} loaded successfully")
    return LOADED_MODELS[model_name]

def models(request):
    """返回可用模型列表"""
    model_list = [{"name": name, "path": path} for name, path in MODELS.items()]
    return JsonResponse({"models": model_list})

@csrf_exempt
def index(request):
    """渲染主页"""
    return render(request, 'fakedetect/index.html')

def build_transform(resize=True, center_crop=True, crop_size=224):
    """构建图像转换管道"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform_steps = []
    if resize:
        transform_steps.append(transforms.Resize((crop_size, crop_size)))
    if center_crop:
        transform_steps.append(transforms.CenterCrop(crop_size))
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(transform_steps)

@csrf_exempt
def predict(request):
    """处理预测请求"""
    start_time = time.time()
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    
    try:
        files = request.FILES.getlist('files')
        if not files:
            return JsonResponse({'error': 'No files uploaded'}, status=400)
            
        model_name = request.POST.get('model')
        if model_name not in MODELS:
            return JsonResponse({'error': f'Model {model_name} not found'}, status=404)
        
        center_crop = request.POST.get('center_crop') == 'true'
        resize = request.POST.get('resize') == 'true'
        crop_size = 224

        # 加载模型
        model = get_or_load_model(model_name)
        
        # 构建转换管道
        transform = build_transform(resize, center_crop, crop_size)

        if len(files) == 1:
            # 单张图片预测
            try:
                img = Image.open(files[0]).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    prediction = model(img_tensor)
                    probability = torch.sigmoid(prediction).item()
                
                processing_time = time.time() - start_time
                logger.info(f"Single image prediction completed in {processing_time:.2f} seconds")
                
                return JsonResponse({
                    'message': f'Prediction: {"Fake" if probability > 0.5 else "Real"}',
                    'chartData': [1 - probability, probability],
                    'processing_time': f"{processing_time:.2f}s"
                })
            except Exception as e:
                return JsonResponse({'error': f'Image processing failed: {str(e)}'}, status=500)
        else:
            # 数据集评估 - 使用批处理
            batch_size = 16  # 可以根据内存调整
            num_files = len(files)
            results = []
            
            try:
                for i in range(0, num_files, batch_size):
                    batch_files = files[i:i+batch_size]
                    batch_tensors = []
                    file_names = []
                    
                    for file in batch_files:
                        img = Image.open(file).convert('RGB')
                        img_tensor = transform(img)
                        batch_tensors.append(img_tensor)
                        file_names.append(file.name)
                    
                    batch_input = torch.stack(batch_tensors)
                    with torch.no_grad():
                        batch_predictions = model(batch_input)
                        batch_probabilities = torch.sigmoid(batch_predictions).squeeze().tolist()
                    
                    # 处理单个元素的情况
                    if not isinstance(batch_probabilities, list):
                        batch_probabilities = [batch_probabilities]
                    
                    # 将文件名和预测结果组合
                    for j, prob in enumerate(batch_probabilities):
                        results.append({
                            'filename': file_names[j],
                            'probability': prob,
                            'prediction': 'Fake' if prob > 0.5 else 'Real',
                            'is_fake': 'fake' in file_names[j].lower()  # 根据文件名判断真实标签
                        })
                
                processing_time = time.time() - start_time
                logger.info(f"Dataset evaluation completed in {processing_time:.2f} seconds for {num_files} images")
                
                return JsonResponse({
                    'results': results,
                    'processing_time': f"{processing_time:.2f}s"
                })
            except Exception as e:
                return JsonResponse({'error': f'Dataset evaluation failed: {str(e)}'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Prediction failed: {str(e)}'}, status=500)
