import lantern
import torch

# convert from lantern tensor to torch tensor
t = lantern.randn((3, 512, 512))
torch_tensor = t.to_torch()
print(torch_tensor)
print(type(torch_tensor))

# convert from torch tensor to lantern tensor
t = torch.randn((3, 512, 512))
lantern_tensor = lantern.Tensor(t)
print(lantern_tensor)
print(type(lantern_tensor))
