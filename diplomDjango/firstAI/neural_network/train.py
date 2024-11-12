# train.py

import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()  # Функция потерь для бинарной классификации
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, num_epochs=10):
        self.model.train()  # Устанавливаем режим обучения
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)  # Преобразуем метки в форму [batch_size, 1]

                # Прямой проход
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Обратное распространение и оптимизация
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Эпоха [{epoch+1}/{num_epochs}], Потеря: {avg_loss:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()  # Устанавливаем режим оценки
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)  # Преобразуем метки в форму [batch_size, 1]
                outputs = self.model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Точность на тестовом наборе: {accuracy:.2f}%')
