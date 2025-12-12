import torch
data = torch.load('world_weights.pt')
print(type(data))
print(data.shape)
print(data.keys() if isinstance(data, dict) else data)
