import torch
# 在loss.backward()之后添加
torch.cuda.empty_cache()