# test_model.py

import torch
from firstAI.neural_network.preprocess import DataLoader
from firstAI.neural_network.AI_model import NeuralNetwork
from firstAI.neural_network.train import Trainer
from firstAI.neural_network.utils import get_data_loader
import numpy as np

def test():
    # Инициализация DataLoader для тестирования
    data_loader = DataLoader(data_source='testing')
    X_test, y_test = data_loader.get_data(limit=100)  # Используем все данные для тестирования

    # Загрузка масштабировщика
    data_loader.load_scaler('scaler.save')

    # Масштабирование тестовых данных
    X_test = data_loader.scale_data(X_test)

    # Подготовка загрузчика данных для тестирования
    test_loader = get_data_loader(X_test, y_test)

    # Определение размера входных данных
    input_size = X_test.shape[1]

    # Инициализация модели
    model = NeuralNetwork(input_size)

    # Загрузка обученной модели
    model.load_state_dict(torch.load('model.pth'))

    # Настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Инициализация тренера
    trainer = Trainer(model, device)

    # Оценка модели
    trainer.evaluate(test_loader)

if __name__ == '__main__':
    test()
