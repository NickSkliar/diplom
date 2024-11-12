# train_model.py

import torch
from firstAI.neural_network.preprocess import DataLoader
from firstAI.neural_network.AI_model import NeuralNetwork
from firstAI.neural_network.train import Trainer
from firstAI.neural_network.utils import get_data_loader

def main():
    # Определение устройства (CPU или GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Инициализация DataLoader для обучения
    data_loader = DataLoader(data_source='training')

    # Загрузка и предобработка данных
    print("Загрузка и предобработка данных...")
    X_train, y_train = data_loader.get_data()

    # Создание DataLoader для обучения
    print("Создание DataLoader...")
    train_loader = get_data_loader(X_train, y_train, batch_size=64)

    # Получение весов признаков
    print("Получение весов признаков...")
    feature_weights = data_loader.get_feature_weights()

    # Инициализация модели
    print("Инициализация модели...")
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size, feature_weights=feature_weights)

    # Инициализация тренера
    print("Инициализация тренера...")
    trainer = Trainer(model, device, learning_rate=0.001)

    # Обучение модели
    print("Начало обучения модели...")
    trainer.train(train_loader, num_epochs=10)

    # Сохранение модели и масштабировщика
    print("Сохранение модели...")
    torch.save(model.state_dict(), 'trained_model.pth')
    data_loader.save_scaler('scaler.pkl')
    print("Модель и масштабировщик успешно сохранены.")

if __name__ == '__main__':
    main()
