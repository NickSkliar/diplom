#firstAI/management/commands/train_model.py

from django.core.management.base import BaseCommand
from firstAI.neural_network.train_model import main

class Command(BaseCommand):
    help = 'Обучение модели нейронной сети'

    def handle(self, *args, **kwargs):
        self.stdout.write('Начало обучения модели...')
        main()
        self.stdout.write(self.style.SUCCESS('Обучение модели успешно завершено.'))
