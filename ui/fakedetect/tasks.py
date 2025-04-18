import os
import io
import time
import json
import logging
import base64
import uuid
from pathlib import Path
import sys

import torch
import torchvision.transforms as transforms # 需要导入 transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from celery import shared_task
from django.conf import settings

# --- 关键：需要将共享的函数移到这里或 utils.py --- 
# 你需要将 views.py 中的 get_model, get_or_load_model, build_transform, extract_features_fallback 
# 以及相关的全局变量（MODELS, DEVICE, LOADED_MODELS）的定义和逻辑移到这里或一个共享的 utils.py 文件中。
# Celery worker 独立运行，不能直接访问 views.py 运行时的内存。

# 示例占位符 (你需要用实际代码替换！)
logger = logging.getLogger(__name__)

# 假设这些从 settings 或 utils 加载
# MODELS = settings.MODELS_DICT 
# DEVICE = settings.DEVICE_SETTING
# LOADED_MODELS = {} # Worker 进程需要自己的缓存

# 示例：需要完整的 get_model 实现 (可能需要从 /root/UniversalFakeDetect/models 导入)
try:
    # 尝试导入原始 get_model (假设路径已在 sys.path 或 PYTHONPATH)
    # 这需要根据您的项目结构调整
    sys.path.append("/root/UniversalFakeDetect") # 如果 Worker 需要
    from models import get_model # 导入原始 get_model
except ImportError:
    logger.error("Failed to import get_model from /root/UniversalFakeDetect/models")
    # 提供一个备用或抛出错误
    def get_model(name): raise NotImplementedError("get_model not available in task environment")

def build_transform(resize=True, center_crop=True, crop_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_steps = []
    if resize: transform_steps.append(transforms.Resize((crop_size, crop_size)))
    if center_crop: transform_steps.append(transforms.CenterCrop(crop_size))
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(transform_steps)

def extract_features_fallback(model, img_tensor, model_name):
    # ... (将 views.py 中的完整 extract_features_fallback 代码复制到这里) ...
    # (代码省略，请从 views.py 复制)
    # 确保此函数内使用的 logger, torch, nn (如果用到了) 都已导入
    pass # Placeholder

def get_or_load_model(model_name):
    # ... (将 views.py 中的 get_or_load_model 完整代码复制到这里) ...
    # (代码省略，请从 views.py 复制，并处理全局 LOADED_MODELS)
    pass # Placeholder

# ------------------------------------------------------

@shared_task(bind=True)
def generate_tsne_task(self, base_dir, relative_file_paths, model_name, perplexity, iterations, pca_components=50):
    task_id = self.request.id
    logger.info(f"Task {task_id}: Starting t-SNE generation for {len(relative_file_paths)} files from base {base_dir}.")
    start_time = time.time()

    # 从 settings 加载配置 (更健壮的方式)
    # MODELS = settings.MODELS_DICT
    # DEVICE = settings.DEVICE_SETTING
    # --- 临时使用全局变量 (需要确保 worker 能访问) --- 
    global MODELS, DEVICE
    if 'MODELS' not in globals() or 'DEVICE' not in globals():
         logger.error(f"Task {task_id}: MODELS or DEVICE not initialized in worker!")
         MODELS = { # 临时的默认值
             "CLIP:ViT-L/14": "/root/autodl-tmp/checkpoints/clip_vitl14/fc_weights.pth",
             "Imagenet:vit_b_16": "/root/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
         }
         DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------

    try:
        processed_files_info = []
        successful_extractions = 0
        batch_size = 32
        total_files = len(relative_file_paths)
        total_batches = (total_files + batch_size - 1) // batch_size
        logger.info(f"Task {task_id}: Extracting features in {total_batches} batches...")
        
        # 加载模型 (确保 get_or_load_model 在此可用)
        model = get_or_load_model(model_name)
        transform = build_transform()

        extraction_start_time = time.time()
        base_path = Path(base_dir)

        for batch_idx in range(total_batches):
            self.update_state(state='PROGRESS', meta={'current': batch_idx + 1, 'total': total_batches, 'step': 'feature_extraction'})
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_files)
            current_batch_relative_paths = relative_file_paths[batch_start:batch_end]
            batch_data_to_add = []

            for relative_path in current_batch_relative_paths:
                try:
                    # 使用基础路径和相对路径构建完整路径
                    img_path = base_path / relative_path 
                    if not img_path.is_file():
                        logger.warning(f"Task {task_id}: File not found at {img_path}, skipping.")
                        continue
                    
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        # 确保 extract_features_fallback 在此可用
                        feature = extract_features_fallback(model, img_tensor, model_name)
                        feature = feature.view(feature.size(0), -1).cpu().numpy()
                    if feature.size == 0 or np.isnan(feature).any() or np.isinf(feature).any():
                        logger.warning(f"Task {task_id}: Invalid feature for {relative_path}, skipping.")
                        continue
                    batch_data_to_add.append((relative_path, feature))
                except Exception as img_ex:
                    logger.error(f"Task {task_id}: Error processing image {relative_path}: {img_ex}")
                    continue
            
            if not batch_data_to_add: continue
            first_shape = batch_data_to_add[0][1].shape
            compatible_batch_data = []
            for path, feat in batch_data_to_add:
                if feat.shape == first_shape: compatible_batch_data.append((path, feat))
                else: logger.warning(f"Task {task_id}: Shape mismatch for {path}")
            if not compatible_batch_data: continue
            processed_files_info.extend(compatible_batch_data)
            successful_extractions += len(compatible_batch_data)
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        logger.info(f"Task {task_id}: Feature extraction finished. Success: {successful_extractions}. Time: {time.time() - extraction_start_time:.2f}s")
        if successful_extractions < 2: raise ValueError(f"Need at least 2 valid images, only processed {successful_extractions}")

        # --- PCA ---
        self.update_state(state='PROGRESS', meta={'step': 'pca'})
        logger.info(f"Task {task_id}: Starting PCA...")
        pca_start_time = time.time()
        features_for_pca = [info[1] for info in processed_files_info]
        features_array = np.vstack(features_for_pca)
        if np.isnan(features_array).any() or np.isinf(features_array).any():
            logger.warning("NaN or Inf detected before PCA. Cleaning."); features_array = np.nan_to_num(features_array)
        n_samples, n_features = features_array.shape
        actual_pca_components = min(pca_components, n_samples - 1, n_features)
        if actual_pca_components < 2: raise ValueError("Not enough data/features for PCA")
        if features_array.shape[1] > actual_pca_components:
            pca = PCA(n_components=actual_pca_components, random_state=0)
            features_array = pca.fit_transform(features_array)
        logger.info(f"Task {task_id}: PCA done. Shape: {features_array.shape}. Time: {time.time() - pca_start_time:.2f}s")

        # --- t-SNE ---
        self.update_state(state='PROGRESS', meta={'step': 'tsne'})
        logger.info(f"Task {task_id}: Starting t-SNE...")
        tsne_start_time = time.time()
        if np.isnan(features_array).any() or np.isinf(features_array).any():
            logger.warning("NaN or Inf detected before t-SNE. Cleaning."); features_array = np.nan_to_num(features_array)
        effective_perplexity = min(perplexity, features_array.shape[0] - 1)
        if effective_perplexity <= 1: effective_perplexity = min(5, features_array.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=0, perplexity=effective_perplexity, n_iter=iterations, learning_rate='auto', init='pca', verbose=0)
        tsne_results = tsne.fit_transform(features_array)
        logger.info(f"Task {task_id}: t-SNE done. Time: {time.time() - tsne_start_time:.2f}s")

        # --- Plotting ---
        self.update_state(state='PROGRESS', meta={'step': 'plotting'})
        logger.info(f"Task {task_id}: Generating plot...")
        plot_start_time = time.time()
        colors = []
        real_color = '#3498db'; fake_color = '#e74c3c'
        file_paths_for_coloring = [info[0] for info in processed_files_info]
        for file_path in file_paths_for_coloring:
            path_lower = file_path.lower().replace('\\', '/')
            if 'fake' in path_lower: colors.append(fake_color)
            else: colors.append(real_color)
        if len(colors) != len(tsne_results): raise ValueError("Mismatch between colors and t-SNE points.")
        real_count = colors.count(real_color); fake_count = colors.count(fake_color)
        logger.info(f"Task {task_id}: Color assignment: {real_count} real, {fake_count} fake.")
        plt.figure(figsize=(12, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.6, s=20, edgecolors='w', linewidths=0.2)
        plt.title(f't-SNE Visualization ({successful_extractions} images)', fontsize=16)
        plt.xlabel('t-SNE dimension 1', fontsize=14); plt.ylabel('t-SNE dimension 2', fontsize=14)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=real_color, markersize=8, label=f'Real ({real_count})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=fake_color, markersize=8, label=f'Fake ({fake_count})')
        ]
        plt.legend(handles=legend_elements, fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(); buf.seek(0)
        image_png_base64 = base64.b64encode(buf.getvalue()).decode('utf-8'); buf.close()
        logger.info(f"Task {task_id}: Plotting done. Time: {time.time() - plot_start_time:.2f}s")
        total_time = time.time() - start_time
        logger.info(f"Task {task_id}: Completed successfully in {total_time:.2f}s")
        return {
            'status': 'SUCCESS',
            'image': image_png_base64,
            'stats': {'total_images': successful_extractions, 'real_images': real_count, 'fake_images': fake_count, 'processing_time': f"{total_time:.2f}s"}
        }
    except Exception as e:
        logger.error(f"Task {task_id}: Failed! Error: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        return {'status': 'FAILURE', 'error': str(e)}
