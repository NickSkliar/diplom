import torch
print("Доступность CUDA: ", torch.cuda.is_available())
print("CUDA версия: ", torch.version.cuda)
print("Устройство: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU недоступен")
