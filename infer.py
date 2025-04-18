import torch
import torchvision.transforms as transforms
from PIL import Image
from models import get_model
from collections import OrderedDict


model_arch = "CLIP:ViT-L/14"
weights_path = "/root/autodl-tmp/checkpoints/clip_vitl14/model.pth"

model = get_model(model_arch)
state_dict = torch.load(weights_path, map_location='cpu')
state_dict = OrderedDict(
    weight=state_dict["model"]["fc.weight"],
    bias=state_dict["model"]["fc.bias"],
)
torch.save(state_dict, "/root/autodl-tmp/checkpoints/clip_vitl14/fc_weights.pth")
breakpoint()
model.fc.load_state_dict(state_dict)
model.eval()


# def image_classification_demo(image_path, model_arch='res50', weights_path='./pretrained_weights/fc_weights.pth'):
#     """
#     图像分类演示：从加载模型到预测结果的完整流程
    
#     参数:
#         image_path: 要分类的图像路径
#         model_arch: 模型架构名称
#         weights_path: 预训练权重路径
#     """
#     # 步骤1: 加载模型
#     print("1. 加载模型")
#     model = get_model(model_arch)
#     state_dict = torch.load(weights_path, map_location='cpu')
#     model.fc.load_state_dict(state_dict)
#     model.eval()
    
#     # 如果有GPU则使用
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     print(f"   - 模型已加载到 {device}")
    
#     # 步骤2: 准备图像
#     print("\n2. 准备图像")
#     img = Image.open(image_path).convert("RGB")
    
#     # 确定归一化参数
#     stat_from = "imagenet" if model_arch.lower().startswith("imagenet") else "clip"
#     MEAN = {
#         "imagenet": [0.485, 0.456, 0.406],
#         "clip": [0.48145466, 0.4578275, 0.40821073]
#     }
#     STD = {
#         "imagenet": [0.229, 0.224, 0.225],
#         "clip": [0.26862954, 0.26130258, 0.27577711]
#     }
    
#     # 图像预处理
#     transform = transforms.Compose([
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
#     ])
#     img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移至设备
    
#     # 步骤3: 进行预测
#     print("\n3. 进行预测")
#     with torch.no_grad():
#         prediction = model(img_tensor)
#         probability = torch.sigmoid(prediction).item()
    
#     # 步骤4: 解释结果
#     print("\n4. 预测结果")
#     print(f"   - 原始预测值: {prediction.item():.4f}")
#     print(f"   - Sigmoid后概率: {probability:.4f}")
    
#     threshold = 0.5  # 默认阈值
#     label = "伪造图像 (Fake)" if probability > threshold else "真实图像 (Real)"
#     confidence = max(probability, 1 - probability) * 100
    
#     print(f"   - 分类结果: {label}")
#     print(f"   - 置信度: {confidence:.2f}%")
    
#     return probability


# # 使用示例
# if __name__ == "__main__":
#     image_classification_demo("path_to_your_image.jpg")
