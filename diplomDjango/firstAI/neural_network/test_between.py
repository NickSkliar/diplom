# firstAI/neural_network/test_between.py

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from firstAI.neural_network.preprocess import DataLoader
from firstAI.neural_network.AI_model import NeuralNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import joblib
import json
import time


def test_model(start_index, end_index, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Використовується пристрій: {device}')

    start_time = time.time()

    scaler_path = os.path.join('models', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Не знайдено файл масштабувальника '{scaler_path}'.")
    scaler = joblib.load(scaler_path)
    print(f"Масштабувальник завантажено з '{scaler_path}'.")

    # Завантажимо повні тестові дані
    data_loader = DataLoader(data_type='test')
    X_test, y_test = data_loader.get_data_full()

    if len(X_test) == 0:
        print("Немає даних для тестування.")
        return

    if end_index > len(X_test):
        raise ValueError(f"Кінцевий індекс ({end_index}) виходить за межі доступних даних ({len(X_test)})")
    if start_index < 0 or start_index >= end_index:
        raise ValueError(f"Некоректно задані індекси: start={start_index}, end={end_index}")

    # Вибираємо потрібний проміжок
    X_test_slice = X_test[start_index:end_index]
    y_test_slice = y_test[start_index:end_index]

    input_size = X_test_slice.shape[1]
    model_path = os.path.join('models', 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Не знайдено навченої моделі '{model_path}'.")
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    X_test_tensor = torch.from_numpy(X_test_slice).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_slice).float().unsqueeze(1).to(device)

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            all_outputs.append(outputs.cpu())
            all_labels.append(batch_y.cpu())

    if all_outputs:
        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)
        predicted = (outputs >= 0.5).float()
        y_true = labels.numpy()
        y_pred = predicted.numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        correct_predictions = int(sum(y_true == y_pred))
        total_errors = int(sum(y_true != y_pred))
        total_rows = int(len(y_true))
        cm = confusion_matrix(y_true, y_pred)
        print(f'Матриця плутанини:\n{cm}')

        print(f'Точність на тестовій вибірці: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Загальна кількість рядків: {total_rows}')
        print(f'Вгадано правильно: {correct_predictions}')
        print(f'Помилки: {total_errors}')

        end_time = time.time()
        test_time = end_time - start_time
        print(f"Час тестування: {test_time:.2f} секунд")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "total_rows": total_rows,
            "correct_predictions": correct_predictions,
            "errors": total_errors,
            "confusion_matrix": cm.tolist(),
            "test_time_seconds": test_time
        }
        os.makedirs('results', exist_ok=True)
        with open('results/test_metrics.json', 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
        print("Метрики тестування збережено в 'results/test_metrics.json'.")
    else:
        print("Немає даних для тестування.")
