from django.core.management.base import BaseCommand
import pandas as pd
import os
from database.models import download_db  # Импорт вашей модели

class Command(BaseCommand):
    help = 'Импорт данных из CSV-файла в базу данных с выводом информации о процессе'

    def handle(self, *args, **options):
        csv_file_path = '../input_data_from_user/largest_dataSet.csv'

        if not os.path.exists(csv_file_path):
            self.stdout.write(self.style.ERROR(f"Файл {csv_file_path} не найден."))
            return

        self.stdout.write("Файл найден, начинается импорт...")

        try:
            chunk_size = 10000  # Размер части, можно адаптировать в зависимости от памяти
            chunks = pd.read_csv(csv_file_path, chunksize=chunk_size)

            for chunk_num, chunk in enumerate(chunks, start=1):
                records_to_insert = []
                for index, row in chunk.iterrows():
                    # Вывод информации о текущей записи
                    self.stdout.write(f"Запись пакета {index + 1} в части {chunk_num}")

                    records_to_insert.append(download_db(
                        flow_id=row['Flow ID'],
                        src_ip=row['Src IP'],
                        src_port=row['Src Port'],
                        dst_ip=row['Dst IP'],
                        dst_port=row['Dst Port'],
                        protocol=row['Protocol'],
                        flow_duration=row['Flow Duration'],
                        tot_fwd_pkts=row['Tot Fwd Pkts'],
                        tot_bwd_pkts=row['Tot Bwd Pkts'],
                        totlen_fwd_pkts=row['TotLen Fwd Pkts'],
                        totlen_bwd_pkts=row['TotLen Bwd Pkts'],
                        fwd_pkt_len_max=row['Fwd Pkt Len Max'],
                        fwd_pkt_len_min=row['Fwd Pkt Len Min'],
                        fwd_pkt_len_mean=row['Fwd Pkt Len Mean'],
                        fwd_pkt_len_std=row['Fwd Pkt Len Std'],
                        bwd_pkt_len_max=row['Bwd Pkt Len Max'],
                        bwd_pkt_len_min=row['Bwd Pkt Len Min'],
                        bwd_pkt_len_mean=row['Bwd Pkt Len Mean'],
                        bwd_pkt_len_std=row['Bwd Pkt Len Std'],
                        flow_byts_per_sec=row['Flow Byts/s'],
                        flow_pkts_per_sec=row['Flow Pkts/s'],
                        flow_iat_mean=row['Flow IAT Mean'],
                        flow_iat_std=row['Flow IAT Std'],
                        flow_iat_max=row['Flow IAT Max'],
                        flow_iat_min=row['Flow IAT Min'],
                        fwd_iat_tot=row['Fwd IAT Tot'],
                        fwd_iat_mean=row['Fwd IAT Mean'],
                        fwd_iat_std=row['Fwd IAT Std'],
                        fwd_iat_max=row['Fwd IAT Max'],
                        fwd_iat_min=row['Fwd IAT Min'],
                        bwd_iat_tot=row['Bwd IAT Tot'],
                        bwd_iat_mean=row['Bwd IAT Mean'],
                        bwd_iat_std=row['Bwd IAT Std'],
                        bwd_iat_max=row['Bwd IAT Max'],
                        bwd_iat_min=row['Bwd IAT Min'],
                        fwd_psh_flags=row['Fwd PSH Flags'],
                        bwd_psh_flags=row['Bwd PSH Flags'],
                        fwd_urg_flags=row['Fwd URG Flags'],
                        bwd_urg_flags=row['Bwd URG Flags'],
                        fwd_header_len=row['Fwd Header Len'],
                        bwd_header_len=row['Bwd Header Len'],
                        fwd_pkts_per_sec=row['Fwd Pkts/s'],
                        bwd_pkts_per_sec=row['Bwd Pkts/s'],
                        pkt_len_min=row['Pkt Len Min'],
                        pkt_len_max=row['Pkt Len Max'],
                        pkt_len_mean=row['Pkt Len Mean'],
                        pkt_len_std=row['Pkt Len Std'],
                        pkt_len_var=row['Pkt Len Var'],
                        fin_flag_cnt=row['FIN Flag Cnt'],
                        syn_flag_cnt=row['SYN Flag Cnt'],
                        rst_flag_cnt=row['RST Flag Cnt'],
                        psh_flag_cnt=row['PSH Flag Cnt'],
                        ack_flag_cnt=row['ACK Flag Cnt'],
                        urg_flag_cnt=row['URG Flag Cnt'],
                        cwe_flag_count=row['CWE Flag Count'],
                        ece_flag_cnt=row['ECE Flag Cnt'],
                        down_up_ratio=row['Down/Up Ratio'],
                        pkt_size_avg=row['Pkt Size Avg'],
                        fwd_seg_size_avg=row['Fwd Seg Size Avg'],
                        bwd_seg_size_avg=row['Bwd Seg Size Avg'],
                        fwd_byts_per_avg=row['Fwd Byts/b Avg'],
                        fwd_pkts_per_avg=row['Fwd Pkts/b Avg'],
                        fwd_blk_rate_avg=row['Fwd Blk Rate Avg'],
                        bwd_byts_per_avg=row['Bwd Byts/b Avg'],
                        bwd_pkts_per_avg=row['Bwd Pkts/b Avg'],
                        bwd_blk_rate_avg=row['Bwd Blk Rate Avg'],
                        subflow_fwd_pkts=row['Subflow Fwd Pkts'],
                        subflow_fwd_byts=row['Subflow Fwd Byts'],
                        subflow_bwd_pkts=row['Subflow Bwd Pkts'],
                        subflow_bwd_byts=row['Subflow Bwd Byts'],
                        init_fwd_win_byts=row['Init Fwd Win Byts'],
                        init_bwd_win_byts=row['Init Bwd Win Byts'],
                        fwd_act_data_pkts=row['Fwd Act Data Pkts'],
                        fwd_seg_size_min=row['Fwd Seg Size Min'],
                        active_mean=row['Active Mean'],
                        active_std=row['Active Std'],
                        active_max=row['Active Max'],
                        active_min=row['Active Min'],
                        idle_mean=row['Idle Mean'],
                        idle_std=row['Idle Std'],
                        idle_max=row['Idle Max'],
                        idle_min=row['Idle Min'],
                        label=row['Label']
                    ))

                # Пакетная вставка записей для оптимизации производительности
                download_db.objects.bulk_create(records_to_insert, batch_size=1000)

            self.stdout.write(self.style.SUCCESS("Импорт завершен успешно."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Произошла ошибка: {e}"))
