# firstAI/neural_network/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from firstAI.neural_network.preprocess import DataLoader
from firstAI.neural_network.AI_model import NeuralNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib  # Для сохранения scaler
import numpy as np

def train_model(num_epochs=10, batch_size=512, chunk_size=50000):
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')

    # Инициализация DataLoader для тренировочных данных
    data_loader = DataLoader(data_type='train', chunk_size=chunk_size)
    feature_weights = data_loader.get_feature_weights()
    feature_names = data_loader.features  # Добавим список признаков
    input_size = data_loader.get_input_size()
    print(f'Количество признаков: {input_size}')
    print(f'Размерность весов: {len(feature_weights)}')

    # Проверка соответствия размерностей
    if input_size != len(feature_weights):
        raise ValueError(f"Несоответствие размерностей: input_size={input_size}, len(feature_weights)={len(feature_weights)}")

    # Создание модели
    model = NeuralNetwork(input_size, feature_weights=feature_weights)
    model = model.to(device)

    # Параметры обучения
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        # Загрузка данных по чанкам
        for X_train_chunk, y_train_chunk in data_loader.get_data():
            # Преобразование в тензоры
            X_train_tensor = torch.from_numpy(X_train_chunk).to(device)
            y_train_tensor = torch.from_numpy(y_train_chunk).unsqueeze(1).to(device)

            # Создание DataLoader для батчевой обработки
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

        if total_samples > 0:
            epoch_loss /= total_samples
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], нет данных для обучения.')

    # Сохранение обученного scaler
    scaler = data_loader.scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Масштабировщик сохранен в 'models/scaler.pkl'.")

    # Оценка модели на тестовых данных
    evaluate_model(model, device, batch_size, chunk_size, feature_names)

    # Сохранение модели
    model_path = os.path.join('models', 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Модель сохранена в {model_path}')

def evaluate_model(model, device, batch_size, chunk_size, feature_names):
    # Инициализация DataLoader для тестовых данных
    try:
        data_loader_test = DataLoader(data_type='test', chunk_size=chunk_size)
    except FileNotFoundError as e:
        print(e)
        return

    model.eval()
    all_outputs = []
    all_labels = []

    data_available = False

    with torch.no_grad():
        for X_test_chunk, y_test_chunk in data_loader_test.get_data():
            data_available = True
            # Преобразование в тензоры
            X_test_tensor = torch.from_numpy(X_test_chunk).to(device)
            y_test_tensor = torch.from_numpy(y_test_chunk).unsqueeze(1).to(device)

            # Создание DataLoader для батчевой обработки
            test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                all_outputs.append(outputs.cpu())
                all_labels.append(batch_y.cpu())

    if data_available and all_outputs:
        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)
        predicted = (outputs >= 0.5).float()
        y_true = labels.numpy()
        y_pred = predicted.numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f'Точность на тестовой выборке: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
    else:
        print("Нет данных для оценки модели.")

    # Вывод финальных весов первого слоя в читаемом формате
    try:
        final_weights = model.layer1[0].weight.data.cpu().numpy()  # Shape: (64, input_size)
        # Агрегируем веса по признакам (например, усредняем абсолютные значения)
        feature_importance = np.mean(np.abs(final_weights), axis=0)  # Shape: (input_size,)

        # Создаем список пар (признак, вес)
        feature_weight_pairs = list(zip(feature_names, feature_importance))

        # Сортируем по убыванию веса
        feature_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        print("\nВеса признаков (от более важного к менее важному):")
        for feature, weight in feature_weight_pairs:
            print(f"{feature} - {weight:.4f}")
    except Exception as e:
        print(f"Ошибка при выводе весов признаков: {e}")
