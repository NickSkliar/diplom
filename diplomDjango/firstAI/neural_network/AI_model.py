# AI_model.py

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, feature_weights=None):
        super(NeuralNetwork, self).__init__()
        # Определение слоев нейронной сети
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 64),  # Первый скрытый слой
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),  # Второй скрытый слой
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output_layer = nn.Linear(32, 1)  # Выходной слой для бинарной классификации

        # Инициализация весов первого слоя с использованием весов признаков
        if feature_weights is not None:
            with torch.no_grad():
                # Преобразуем список весов в тензор
                feature_weights_tensor = torch.tensor(feature_weights, dtype=torch.float32)
                # Проверяем размерность тензора весов
                if feature_weights_tensor.shape[0] != input_size:
                    raise ValueError("Размерность feature_weights должна соответствовать input_size")
                # Создаем тензор начальных весов для первого слоя
                # Размерность весов первого слоя: [выходные нейроны, входные нейроны]
                # Поэтому транспонируем тензор
                init_weights = feature_weights_tensor.unsqueeze(1).repeat(1, 64).t()
                # Устанавливаем веса первого слоя
                self.layer1[0].weight.data.copy_(init_weights)
                # Инициализируем смещение (bias) нулями или как вам необходимо
                self.layer1[0].bias.data.fill_(0.0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.sigmoid(self.output_layer(x))
        return x
