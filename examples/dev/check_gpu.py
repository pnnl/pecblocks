import torch

print (torch.cuda.is_available())
devices = [d for d in range(torch.cuda.device_count())]
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print (device_names)
