import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")

print("----------check torch----------")
import torch
print(f"torch version: {torch.version.cuda}")

print("----------check onnx----------")
import onnxruntime as ort
# 列出可用的执行提供者
print(ort.get_available_providers())
# 输出应包含 CUDA 相关的提供者（如 'CUDAExecutionProvider'）
# 同时验证 CUDA 版本
providers = ort.get_device()
print("Device:", providers)  # 应显示 'GPU'
# 查看 ONNX Runtime 的详细信息
print(ort.__version__)
print(ort.get_build_info())

print("----------转换代码----------")
from onnx2pytorch import ConvertModel
import torch
import onnx

# 加载 ONNX 模型
onnx_model = onnx.load("1.onnx")

# 转换为 PyTorch 模型
pytorch_model = ConvertModel(onnx_model)

# 保存为 PT 文件
torch.save(pytorch_model.state_dict(), "s1.pt")  # 仅保存权重
# 或保存整个模型（包括结构）
# torch.save(pytorch_model, "model.pth")