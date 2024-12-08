# firstAI/management/commands/test_model.py

from django.core.management.base import BaseCommand
from firstAI.neural_network.test_between import test_model

class Command(BaseCommand):
    help = 'Тестування нейронної мережі на вказаному проміжку даних'

    def add_arguments(self, parser):
        parser.add_argument('--start', type=int, required=True, help='Початковий індекс')
        parser.add_argument('--end', type=int, required=True, help='Кінцевий індекс')
        parser.add_argument('--batch_size', type=int, default=512, help='Розмір батчу')

    def handle(self, *args, **options):
        start_index = options['start']
        end_index = options['end']
        batch_size = options['batch_size']

        # Викликаємо функцію test_model з передачею аргументів
        test_model(start_index=start_index, end_index=end_index, batch_size=batch_size)

# Команда для запуску (приклад):
# python manage.py test_model --start 1000 --end 2000 --batch_size 256
