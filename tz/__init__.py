import torch

# bf16 (what Qwen3 was trained in) on Ampere+ GPUs (A100, L4); fp16 on older ones like the T4.
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
