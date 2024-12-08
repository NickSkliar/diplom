# firstAI/neural_network/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from firstAI.neural_network.preprocess import DataLoader
from firstAI.neural_network.AI_model import NeuralNetwork
import os
import joblib
import numpy as np
import json
import time

def train_model(num_epochs=10, batch_size=512, chunk_size=50000):
    # Перевіряємо доступність GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Використовується пристрій: {device}')

    # Ініціалізація DataLoader для тренувальних даних
    data_loader = DataLoader(data_type='train', chunk_size=chunk_size)
    feature_weights = data_loader.get_feature_weights()
    feature_names = data_loader.features
    input_size = data_loader.get_input_size()

    # Перевірка відповідності розмірностей
    if input_size != len(feature_weights):
        raise ValueError(f"Невідповідність розмірностей: input_size={input_size}, len(feature_weights)={len(feature_weights)}")

    # Створення моделі
    model = NeuralNetwork(input_size, feature_weights=feature_weights)
    model = model.to(device)

    # Параметри навчання
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Для збереження результатів навчання
    training_results = {"losses": [], "feature_importance": [], "training_time_seconds": None}

    # Засікаємо час навчання
    start_time = time.time()

    # Навчання моделі
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        # Завантаження даних по чанках
        for X_train_chunk, y_train_chunk in data_loader.get_data():
            X_train_tensor = torch.from_numpy(X_train_chunk).float().to(device)
            y_train_tensor = torch.from_numpy(y_train_chunk).float().unsqueeze(1).to(device)

            # Створення DataLoader для пакетної обробки
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
            training_results["losses"].append({"epoch": epoch + 1, "loss": epoch_loss})
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], немає даних для навчання.')

    # Вимірюємо час навчання
    end_time = time.time()
    training_time = end_time - start_time
    training_results["training_time_seconds"] = training_time
    print(f"Час навчання: {training_time:.2f} секунд")

    # Збереження навченого масштабувальника
    scaler = data_loader.scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Масштабувальник збережено в 'models/scaler.pkl'.")

    # Збереження моделі
    model_path = os.path.join('models', 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Модель збережено в {model_path}')

    # Виведення та збереження фінальних ваг першого шару
    try:
        if hasattr(model, 'layer1') and isinstance(model.layer1, nn.Sequential):
            final_weights = model.layer1[0].weight.data.cpu().numpy()
        else:
            final_weights = model.fc1.weight.data.cpu().numpy()  # Наприклад, якщо шар називається fc1

        feature_importance = np.mean(np.abs(final_weights), axis=0)
        feature_weight_pairs = list(zip(feature_names, feature_importance))
        feature_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        print("\nВаги ознак (від найважливішої до найменш важливої):")
        for feature, weight in feature_weight_pairs:
            print(f"{feature} - {weight:.4f}")
            training_results["feature_importance"].append({
                "feature": feature,
                "weight": float(weight)
            })

        # Збереження результатів у JSON
        with open('models/training_results.json', 'w') as json_file:
            json.dump(training_results, json_file, indent=4)
        print("Результати навчання збережено в 'models/training_results.json'.")
    except Exception as e:
        print(f"Помилка при виведенні ваг ознак: {e}")
