import torch

# bf16 (what Qwen3 was trained in) on Ampere+ GPUs (A100, L4); fp16 on older ones like the T4.
# The capability query can raise before a GPU is attached (ZeroGPU patches cuda.is_available()
# to True at import but attaches the GPU only inside @spaces.GPU), so assume a modern GPU then.
try:
    _bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
except Exception:
    _bf16 = torch.cuda.is_available()
DTYPE = torch.bfloat16 if _bf16 else torch.float16
