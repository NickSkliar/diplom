# firstAI/neural_network/AI_model.py

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, feature_weights=None):
        super(NeuralNetwork, self).__init__()
        # Определение слоев нейронной сети
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 40),  # Перший скритий шар
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(40, 25),  # 2 скритий шар
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(25, 13),  # 3 скритий шар
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output_layer = nn.Linear(13, 1)  # Выходной слой для бинарной классификации

        # Инициализация весов первого слоя с использованием весов признаков
        if feature_weights is not None:
            with torch.no_grad():
                feature_weights_tensor = torch.tensor(feature_weights, dtype=torch.float32)
                if feature_weights_tensor.shape[0] != input_size:
                    raise ValueError("Розмірність feature_weights має відповідати input_size")
                init_weights = self.layer1[0].weight.data
                init_weights = init_weights * feature_weights_tensor.unsqueeze(0)
                self.layer1[0].weight.data.copy_(init_weights)
                self.layer1[0].bias.data.fill_(0.0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.sigmoid(self.output_layer(x))
        return x
