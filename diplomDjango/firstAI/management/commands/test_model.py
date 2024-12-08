# firstAI/management/commands/test_model.py

from django.core.management.base import BaseCommand
from firstAI.neural_network.test import test_model

class Command(BaseCommand):
    help = 'Тестирование нейронной сети'

    def add_arguments(self, parser):
        parser.add_argument('--batch_size', type=int, default=512, help='Размер батча')
        parser.add_argument('--chunk_size', type=int, default=50000, help='Размер чанка для загрузки данных')

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        chunk_size = options['chunk_size']

        # Вызываем функцию test_model
        test_model(batch_size=batch_size, chunk_size=chunk_size)
