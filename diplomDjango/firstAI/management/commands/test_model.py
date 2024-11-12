# firstAI/management/commands/test_model.py

from django.core.management.base import BaseCommand
from firstAI.neural_network.test_model import test

class Command(BaseCommand):
    help = 'Тестирование модели нейронной сети'

    def handle(self, *args, **kwargs):
        try:
            self.stdout.write('Начало тестирования модели...')
            test()
            self.stdout.write(self.style.SUCCESS('Тестирование модели успешно завершено.'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Ошибка во время тестирования: {e}'))
