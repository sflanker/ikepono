import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available')
elif torch.backends.mps.is_built():
    device = torch.device('cuda:0')
    print('MPS is available')
else:
    device = torch.device('cpu')
    print('CUDA and MPS are not available')