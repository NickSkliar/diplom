from django.core.management.base import BaseCommand
import pandas as pd
import os
from database.models import DownloadDB  # Імпорт моделі
import datetime
import pytz  # Для роботи з часовими поясами

class Command(BaseCommand):
    help = 'Імпорт даних з CSV-файлу в модель DownloadDB з обробкою часових поясів'

    def handle(self, *args, **options):
        csv_file_path = '../input_data_from_user/cleaned_file.csv'

        if not os.path.exists(csv_file_path):
            self.stdout.write(self.style.ERROR(f"Файл {csv_file_path} не знайдено."))
            return

        self.stdout.write("Файл знайдено, починається імпорт...")

        try:
            chunk_size = 1000000  # Розмір частини, можна адаптувати в залежності від пам'яті
            chunks = pd.read_csv(csv_file_path, chunksize=chunk_size)

            total_rows = 0
            skipped_rows = 0
            total_records_inserted = 0
            timezone = pytz.UTC  # Встановлення часового поясу в UTC, змініть за потребою

            for chunk_num, chunk in enumerate(chunks, start=1):
                records_to_insert = []
                chunk_rows = 0  # Кількість рядків у поточному чанку
                chunk_skipped = 0  # Кількість пропущених рядків у поточному чанку
                for index, row in chunk.iterrows():
                    total_rows += 1
                    chunk_rows += 1
                    try:
                        # Перевіряємо наявність та непорожність поля 'Timestamp'
                        timestamp_str = row.get('Timestamp')
                        if pd.isnull(timestamp_str) or timestamp_str == '':
                            skipped_rows += 1
                            chunk_skipped += 1
                            # Виводимо причину пропуску рядка
                            self.stdout.write(f"Рядок {index + 1} пропущено: відсутнє або порожнє поле 'Timestamp'.")
                            continue
                        # Список можливих форматів дати і часу
                        timestamp_formats = [
                            '%d/%m/%Y %I:%M:%S %p',  # Формат з AM/PM
                            '%d/%m/%Y %H:%M:%S',     # Формат з 24-годинним часом
                            '%Y-%m-%d %H:%M:%S',     # Формат '2010-06-12 08:34:32'
                        ]
                        for fmt in timestamp_formats:
                            try:
                                timestamp = datetime.datetime.strptime(timestamp_str, fmt)
                                # Якщо успішно розпарсено, виходимо з циклу
                                break
                            except ValueError:
                                timestamp = None
                                continue
                        if timestamp is None:
                            # Пропускаємо рядок, якщо жоден формат не підійшов
                            skipped_rows += 1
                            chunk_skipped += 1
                            self.stdout.write(f"Рядок {index + 1} пропущено: неправильний формат 'Timestamp'.")
                            continue
                        # Застосування часового поясу після успішного парсингу
                        timestamp = timezone.localize(timestamp)

                        # Список обов'язкових полів
                        required_fields = {
                            'src_ip': row.get('Src IP', ''),
                            'dst_ip': row.get('Dst IP', ''),
                            'src_port': row.get('Src Port', ''),
                            'dst_port': row.get('Dst Port', ''),
                            'protocol': row.get('Protocol', ''),
                            'flow_duration': row.get('Flow Duration', ''),
                            'tot_fwd_pkts': row.get('Tot Fwd Pkts', ''),
                            'tot_bwd_pkts': row.get('Tot Bwd Pkts', ''),
                            'flow_byts_per_sec': row.get('Flow Byts/s', ''),
                            'flow_pkts_per_sec': row.get('Flow Pkts/s', ''),
                            'fwd_iat_mean': row.get('Fwd IAT Mean', ''),
                            'bwd_iat_mean': row.get('Bwd IAT Mean', ''),
                            'down_up_ratio': row.get('Down/Up Ratio', ''),
                            'pkt_size_avg': row.get('Pkt Size Avg', ''),
                            'fwd_pkts_per_sec': row.get('Fwd Pkts/s', ''),
                            'bwd_pkts_per_sec': row.get('Bwd Pkts/s', ''),
                            'label': row.get('Label', ''),
                        }

                        # Перевіряємо, чи є порожні або відсутні значення в обов'язкових полях
                        missing_fields = [key for key, value in required_fields.items() if pd.isnull(value) or value == '']
                        if missing_fields:
                            skipped_rows += 1
                            chunk_skipped += 1
                            self.stdout.write(f"Рядок {index + 1} пропущено: відсутні обов'язкові поля {', '.join(missing_fields)}.")
                            continue

                        # Створення екземпляра моделі DownloadDB
                        record = DownloadDB(
                            src_ip=required_fields['src_ip'],
                            dst_ip=required_fields['dst_ip'],
                            src_port=int(required_fields['src_port']),
                            dst_port=int(required_fields['dst_port']),
                            protocol=int(required_fields['protocol']),
                            timestamp=timestamp,
                            flow_duration=int(required_fields['flow_duration']),
                            tot_fwd_pkts=int(required_fields['tot_fwd_pkts']),
                            tot_bwd_pkts=int(required_fields['tot_bwd_pkts']),
                            flow_byts_per_sec=float(required_fields['flow_byts_per_sec']),
                            flow_pkts_per_sec=float(required_fields['flow_pkts_per_sec']),
                            fwd_iat_mean=float(required_fields['fwd_iat_mean']),
                            bwd_iat_mean=float(required_fields['bwd_iat_mean']),
                            down_up_ratio=float(required_fields['down_up_ratio']),
                            pkt_size_avg=float(required_fields['pkt_size_avg']),
                            fwd_pkts_per_sec=float(required_fields['fwd_pkts_per_sec']),
                            bwd_pkts_per_sec=float(required_fields['bwd_pkts_per_sec']),
                            label=required_fields['label'],
                            data_type='unlabeled'  # Встановлюємо значення за замовчуванням
                        )

                        records_to_insert.append(record)
                    except Exception as e:
                        skipped_rows += 1
                        chunk_skipped += 1
                        self.stdout.write(f"Рядок {index + 1} пропущено: помилка при обробці ({e}).")
                        continue

                # Пакетне вставлення записів для оптимізації продуктивності
                DownloadDB.objects.bulk_create(records_to_insert, batch_size=200000)
                records_inserted = len(records_to_insert)
                total_records_inserted += records_inserted

                # Відображення кількості записаних записів після вставки
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Частина {chunk_num} завершена: оброблено рядків {chunk_rows}, записано {records_inserted} записів, "
                        f"пропущено у цій частині {chunk_skipped} рядків."
                    )
                )

            self.stdout.write(
                self.style.SUCCESS(
                    f"Імпорт успішно завершено. Всього рядків: {total_rows}, всього записано: {total_records_inserted}, "
                    f"всього пропущено: {skipped_rows}"
                )
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Сталася помилка: {e}"))
