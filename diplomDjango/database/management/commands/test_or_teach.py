from django.core.management.base import BaseCommand
from database.models import DownloadDB
from django.db.models.functions import TruncDate
from django.db.models import Count
from datetime import datetime

class Command(BaseCommand):
    help = 'Розподіляє дані на навчальні та тестові за принципом: 2 дні на навчання, 1 день на тестування.'

    def handle(self, *args, **options):
        # Отримуємо унікальні дати з бази даних та сортуємо їх
        dates = DownloadDB.objects.annotate(date=TruncDate('timestamp')) \
            .values('date') \
            .distinct() \
            .order_by('date')

        # Перетворюємо їх у список дат
        date_list = [item['date'] for item in dates]

        # Ініціалізуємо лічильники
        train_days = 2
        test_days = 1
        total_days = len(date_list)
        index = 0

        while index < total_days:
            # Встановлюємо дні для навчання
            train_dates = date_list[index:index+train_days]
            for train_date in train_dates:
                updated = DownloadDB.objects.filter(timestamp__date=train_date).update(data_type='train')
                self.stdout.write(self.style.SUCCESS(
                    f"Дата {train_date.strftime('%Y-%m-%d')} встановлена як НАВЧАННЯ. Оновлено {updated} записів."
                ))
            index += train_days

            # Встановлюємо дні для тестування
            test_dates = date_list[index:index+test_days]
            for test_date in test_dates:
                updated = DownloadDB.objects.filter(timestamp__date=test_date).update(data_type='test')
                self.stdout.write(self.style.SUCCESS(
                    f"Дата {test_date.strftime('%Y-%m-%d')} встановлена як ТЕСТУВАННЯ. Оновлено {updated} записів."
                ))
            index += test_days

        # Перевірка на нерозмічені дані
        unlabeled_count = DownloadDB.objects.filter(data_type='unlabeled').count()
        if unlabeled_count > 0:
            self.stdout.write(self.style.WARNING(f'Залишилося {unlabeled_count} нерозмічених записів.'))
        else:
            self.stdout.write(self.style.SUCCESS('Усі записи успішно розмічені.'))
