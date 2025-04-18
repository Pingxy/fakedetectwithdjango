import os
import json
import time
import logging
import uuid
import tempfile
from pathlib import Path

from django.http import JsonResponse, Http404
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings # 导入 Django settings

from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import base64

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django import forms

from .models import CustomUser, DetectionRecord
from .tasks import generate_tsne_task # 导入 Celery 任务
from celery.result import AsyncResult # 导入 AsyncResult

import sys
sys.path.append("/root/UniversalFakeDetect")
from models import get_model

import numpy as np

# 自定义用户创建表单
class CustomUserCreationForm(forms.ModelForm):
    password1 = forms.CharField(label='密码', widget=forms.PasswordInput)
    password2 = forms.CharField(label='确认密码', widget=forms.PasswordInput)

    class Meta:
        model = CustomUser
        fields = ('username',)

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("两次密码输入不一致")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user


# 设置日志
logger = logging.getLogger(__name__)

# --- 注意：以下全局变量和函数最好移到 utils.py 或 settings.py --- 
# 模型字典 (应从 settings 加载)
MODELS = {
    "CLIP:ViT-L/14": "/root/autodl-tmp/checkpoints/clip_vitl14/fc_weights.pth",
    "Imagenet:vit_b_16": "/root/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
}
# 全局变量缓存已加载的模型 (线程不安全, 生产环境可能有问题)
LOADED_MODELS = {}
# 检查CUDA是否可用 (应从 settings 加载)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------------------------------

def extract_features_fallback(model, img_tensor, model_name):
    """Fallback feature extraction method"""
    try:
        if hasattr(model, 'get_features'):
             logger.info("Using model.get_features() from wrapper")
             return model.get_features(img_tensor)
        if model_name.startswith("CLIP:") and hasattr(model, 'model') and hasattr(model.model, 'encode_image'):
            logger.info("Using model.model.encode_image() from CLIPModel wrapper")
            features = model.model.encode_image(img_tensor)
            return features
        if hasattr(model, 'features'):
            logger.info("Using model.features()")
            features = model.features(img_tensor)
            return features
        if hasattr(model, 'backbone'):
            logger.info("Using model.backbone()")
            return model.backbone(img_tensor)
        if hasattr(model, 'fc'):
            logger.info("Using model and removing final fc layer")
            original_fc = model.fc
            try:
                model.fc = torch.nn.Identity()
                features = model(img_tensor)
            finally:
                model.fc = original_fc 
            return features
        logger.warning(f"No standard feature extraction method found for {model_name}. Falling back to direct model call, results may be classification logits.")
        return model(img_tensor)
    except Exception as ex:
        logger.error(f"Error during feature extraction fallback: {str(ex)}", exc_info=True)
        logger.warning(f"Exception during fallback for {model_name}. Trying direct model call as last resort.")
        return model(img_tensor)

def get_or_load_model(model_name):
    """获取已加载的模型或加载新模型 (非线程安全) """
    # ！！ 警告：这个全局缓存不是线程安全的，在多线程/多进程服务器中会出问题
    # ！！ 生产环境需要更健壮的模型加载和缓存策略
    global LOADED_MODELS, MODELS, DEVICE # 显式引用全局变量
    if model_name not in LOADED_MODELS:
        logger.info(f"Loading model: {model_name}")
        model_instance = get_model(model_name) # 假设 get_model 从外部导入
        state_dict = torch.load(MODELS[model_name], map_location=DEVICE)
        model_instance.fc.load_state_dict(state_dict)
        model_instance = model_instance.to(DEVICE)
        model_instance.eval()
        LOADED_MODELS[model_name] = model_instance
        logger.info(f"Model {model_name} loaded successfully on {DEVICE}")
    return LOADED_MODELS[model_name]

def models(request):
    """返回可用模型列表"""
    # 从 settings 加载模型列表更佳
    global MODELS
    model_list = [{"name": name, "path": path} for name, path in MODELS.items()]
    return JsonResponse({"models": model_list})

@login_required
def index(request):
    """渲染主页"""
    return render(request, 'fakedetect/index.html')

def build_transform(resize=True, center_crop=True, crop_size=224):
    """构建图像转换管道"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_steps = []
    if resize: transform_steps.append(transforms.Resize((crop_size, crop_size)))
    if center_crop: transform_steps.append(transforms.CenterCrop(crop_size))
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(transform_steps)

@login_required
@csrf_exempt
def predict(request):
    """处理预测请求 (同步) """
    start_time = time.time()
    if request.method != 'POST': return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    try:
        files = request.FILES.getlist('files')
        if not files: return JsonResponse({'error': 'No files uploaded'}, status=400)
        model_name = request.POST.get('model')
        global MODELS # 使用全局 MODELS (非最佳)
        if model_name not in MODELS: return JsonResponse({'error': f'Model {model_name} not found'}, status=404)
        center_crop = request.POST.get('center_crop') == 'true'
        resize = request.POST.get('resize') == 'true'
        crop_size = 224
        model = get_or_load_model(model_name)
        transform = build_transform(resize, center_crop, crop_size)
        if len(files) == 1:
            try:
                img = Image.open(files[0]).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    prediction = model(img_tensor)
                    probability = torch.sigmoid(prediction).item()
                processing_time = time.time() - start_time
                logger.info(f"Single image prediction completed in {processing_time:.2f} seconds")
                return JsonResponse({'message': f'Prediction: {"Fake" if probability > 0.5 else "Real"}', 'chartData': [1 - probability, probability], 'processing_time': f"{processing_time:.2f}s"})
            except Exception as e: return JsonResponse({'error': f'Image processing failed: {str(e)}'}, status=500)
        else:
            batch_size = 16; num_files = len(files); results = []
            try:
                for i in range(0, num_files, batch_size):
                    batch_files = files[i:i+batch_size]; batch_tensors = []; file_names = []
                    for file in batch_files:
                        img = Image.open(file).convert('RGB'); img_tensor = transform(img)
                        batch_tensors.append(img_tensor); file_names.append(file.name)
                    batch_input = torch.stack(batch_tensors).to(DEVICE)
                    with torch.no_grad():
                        batch_features = extract_features_fallback(model, batch_input, model_name)
                        if hasattr(model, 'fc'): batch_predictions = model.fc(batch_features)
                        else: logger.warning(f"Model {model_name} might not have an fc layer for prediction after fallback."); batch_predictions = batch_features
                        batch_probabilities = torch.sigmoid(batch_predictions).squeeze().cpu().tolist()
                    if not isinstance(batch_probabilities, list): batch_probabilities = [batch_probabilities]
                    for j, prob in enumerate(batch_probabilities):
                        results.append({'filename': file_names[j], 'probability': prob, 'prediction': 'Fake' if prob > 0.5 else 'Real', 'is_fake': 'fake' in file_names[j].lower()})
                processing_time = time.time() - start_time
                logger.info(f"Dataset evaluation completed in {processing_time:.2f} seconds for {num_files} images")
                DetectionRecord.objects.create(user=request.user, result=json.dumps({'results': results, 'processing_time': f"{processing_time:.2f}s", 'model_used': model_name}))
                return JsonResponse({'results': results, 'processing_time': f"{processing_time:.2f}s"})
            except Exception as e: return JsonResponse({'error': f'Dataset evaluation failed: {str(e)}'}, status=500)
    except Exception as e: return JsonResponse({'error': f'Prediction failed: {str(e)}'}, status=500)

@csrf_exempt
@login_required
def tsne_visualization(request):
    """接收 t-SNE 请求，保存文件，触发异步任务并返回任务 ID。"""
    if request.method != 'POST': return JsonResponse({'error': 'Only POST method is supported'}, status=405)
    try:
        start_time = time.time()
        files = request.FILES.getlist('files')
        if not files or len(files) < 2: return JsonResponse({'error': 'Please upload at least 2 images'}, status=400)
        model_name = request.POST.get('model')
        perplexity = int(request.POST.get('perplexity', 30))
        iterations = int(request.POST.get('iterations', 1000))
        # 使用 settings 中的模型字典更佳
        global MODELS
        if model_name not in MODELS: return JsonResponse({'error': f'Model {model_name} not found'}, status=404)
        upload_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.gettempdir()) / 'tsne_uploads' / upload_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(files)} files to temporary directory: {temp_dir}")
        saved_file_paths = []
        for uploaded_file in files:
            file_ext = Path(uploaded_file.name).suffix
            save_name = f"{uuid.uuid4()}{file_ext}"
            # --- 存储相对路径以供任务使用和着色 --- 
            relative_path = getattr(uploaded_file, 'webkitRelativePath', uploaded_file.name)
            if not relative_path: relative_path = uploaded_file.name
            # 在临时目录中创建子目录结构 (如果 webkitRelativePath 可用)
            full_save_path = temp_dir
            relative_dir = Path(relative_path).parent
            if relative_dir and str(relative_dir) != '.':
                 full_save_path = temp_dir / relative_dir
                 full_save_path.mkdir(parents=True, exist_ok=True)
            save_path_obj = full_save_path / save_name
            # -------------------------------------------
            try:
                with open(save_path_obj, 'wb+') as destination:
                    for chunk in uploaded_file.chunks(): destination.write(chunk)
                # 传递相对路径给 Celery 任务，让任务自己解析绝对路径或处理
                saved_file_paths.append(relative_path) # 传递相对路径
            except Exception as save_ex:
                logger.error(f"Error saving file {uploaded_file.name} to {save_path_obj}: {save_ex}")
        if len(saved_file_paths) < 2:
             logger.error("Failed to save enough files for processing.")
             # 清理已保存的文件和目录 (需要实现清理逻辑)
             return JsonResponse({'error': 'Failed to save uploaded files.'}, status=500)
        logger.info(f"Successfully saved {len(saved_file_paths)} files relative paths for task.")
        # --- 触发 Celery 任务 --- 
        # 传递临时目录的基础路径和相对路径列表给任务
        task = generate_tsne_task.delay(
            base_dir=str(temp_dir.resolve()), # 传递临时目录绝对路径
            relative_file_paths=saved_file_paths, # 传递相对路径列表
            model_name=model_name,
            perplexity=perplexity,
            iterations=iterations
        )
        logger.info(f"Dispatched t-SNE task {task.id} in {time.time() - start_time:.2f}s")
        return JsonResponse({'task_id': task.id, 'message': 't-SNE task started.'})
    except Exception as e:
        logger.error(f"Error dispatching t-SNE task: {e}", exc_info=True)
        return JsonResponse({'error': f'Failed to start t-SNE task: {str(e)}'}, status=500)

def get_task_status(request, task_id):
    """根据任务 ID 查询 Celery 任务的状态和结果。"""
    task_result = AsyncResult(task_id)
    response_data = {'task_id': task_id, 'status': task_result.status, 'result': None, 'info': None}
    if task_result.successful(): response_data['result'] = task_result.get()
    elif task_result.failed():
        try: error_info = str(task_result.info) if task_result.info else 'Task failed without details.'; response_data['info'] = {'error': error_info}
        except Exception: response_data['info'] = {'error': 'Task failed, unable to retrieve details.'}
    elif task_result.state == 'PROGRESS': response_data['info'] = task_result.info
    elif task_result.state == 'PENDING': response_data['info'] = {'message': 'Task is waiting to be processed.'}
    elif task_result.state == 'STARTED': response_data['info'] = {'message': 'Task has started processing.'}
    return JsonResponse(response_data)

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(); login(request, user); return redirect('index')
    else: form = CustomUserCreationForm()
    return render(request, 'fakedetect/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated: return redirect('index')
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid(): user = form.get_user(); login(request, user); return redirect('index')
    else: form = AuthenticationForm()
    return render(request, 'fakedetect/login.html', {'form': form})

def logout_view(request):
    logout(request); return redirect('login')

@login_required
def history(request):
    """显示用户历史检测记录"""
    records = DetectionRecord.objects.filter(user=request.user).order_by('-detection_time')
    processed_records = []
    for record in records:
        try:
            result_data = json.loads(record.result)
            results = result_data.get('results', []); total_count = len(results)
            real_count = sum(1 for r in results if r.get('prediction') == 'Real')
            fake_count = total_count - real_count
            processed_records.append({'id': record.id, 'detection_time': record.detection_time, 'model_used': result_data.get('model_used', '未知'), 'processing_time': result_data.get('processing_time', '未知'), 'total_images': total_count, 'real_count': real_count, 'fake_count': fake_count, 'raw_data': record.result})
        except json.JSONDecodeError:
            processed_records.append({'id': record.id, 'detection_time': record.detection_time, 'error': '数据格式错误', 'raw_data': record.result})
    return render(request, 'fakedetect/history.html', {'records': processed_records})

@login_required
def record_detail(request, record_id):
    """显示单个检测记录的详细信息"""
    try:
        record = DetectionRecord.objects.get(id=record_id, user=request.user)
        result_data = json.loads(record.result)
        return render(request, 'fakedetect/record_detail.html', {'record': record, 'result_data': result_data})
    except DetectionRecord.DoesNotExist: return redirect('history')
    except json.JSONDecodeError: return render(request, 'fakedetect/record_detail.html', {'record': record, 'error': '数据格式错误'})
