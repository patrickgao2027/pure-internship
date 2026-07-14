import torch
print("Can Python use the GPU?:", torch.cuda.is_available())
print("CUDA Version Python is using:", torch.version.cuda)

import cuml
print("cuML version:", cuml.__version__)