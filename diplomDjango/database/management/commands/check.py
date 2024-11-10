from django.core.management.base import BaseCommand
import os
import sys
import django

class Command(BaseCommand):
    help = 'Проверка всех подключений и импортов'

    def handle(self, *args, **options):
        # Проверка корректного рабочего каталога
        self.stdout.write(f"Текущий рабочий каталог: {os.getcwd()}")

        # Проверка переменной окружения DJANGO_SETTINGS_MODULE
        if 'DJANGO_SETTINGS_MODULE' not in os.environ:
            self.stdout.write(self.style.ERROR("Переменная DJANGO_SETTINGS_MODULE не установлена."))
            return
        else:
            self.stdout.write(self.style.SUCCESS(f"DJANGO_SETTINGS_MODULE установлен: {os.environ['DJANGO_SETTINGS_MODULE']}"))

        try:
            # Инициализация Django
            django.setup()
            self.stdout.write(self.style.SUCCESS("Django инициализирован успешно."))

            # Проверка импорта модели
            from database.models import download_db
            self.stdout.write(self.style.SUCCESS("Импорт модели выполнен успешно."))

            # Проверка пути к файлу
            csv_file_path = '../input_data_from_user/largest_dataSet.csv'
            if not os.path.exists(csv_file_path):
                self.stdout.write(self.style.ERROR(f"Файл {csv_file_path} не найден."))
            else:
                self.stdout.write(self.style.SUCCESS(f"Файл {csv_file_path} найден."))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Произошла ошибка: {e}"))
