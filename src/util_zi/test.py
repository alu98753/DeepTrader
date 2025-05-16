import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA version (compiled with): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Active CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")