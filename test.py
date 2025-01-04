import torch
print("CUDA Available: ", torch.cuda.is_available())

import dlib
print("Dlib Compiled with CUDA: ", dlib.DLIB_USE_CUDA)



