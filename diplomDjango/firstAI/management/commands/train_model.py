# firstAI/management/commands/train_model.py

from django.core.management.base import BaseCommand
from firstAI.neural_network.train import train_model

class Command(BaseCommand):
    help = 'Обучение нейронной сети для классификации атак'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=10, help='Количество эпох для обучения')
        parser.add_argument('--batch_size', type=int, default=512, help='Размер батча')
        parser.add_argument('--chunk_size', type=int, default=50000, help='Размер чанка для загрузки данных')

    def handle(self, *args, **options):
        num_epochs = options['epochs']
        batch_size = options['batch_size']
        chunk_size = options['chunk_size']

        # Вызываем функцию train_model из модуля train.py
        train_model(num_epochs=num_epochs, batch_size=batch_size, chunk_size=chunk_size)
