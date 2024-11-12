from django.db import models

class download_db(models.Model):
    id = models.AutoField(primary_key=True)
    flow_id = models.CharField(max_length=255, default='', null=True, blank=True)  # Уникальный идентификатор потока
    src_ip = models.GenericIPAddressField(default='0.0.0.0', null=True, blank=True)  # IP-адрес источника
    src_port = models.IntegerField(default=0, null=True, blank=True)  # Порт источника
    dst_ip = models.GenericIPAddressField(default='0.0.0.0', null=True, blank=True)  # IP-адрес назначения
    dst_port = models.IntegerField(default=0, null=True, blank=True)  # Порт назначения
    protocol = models.IntegerField(default=0, null=True, blank=True)  # Протокол передачи данных
    flow_duration = models.BigIntegerField(default=0, null=True, blank=True)  # Длительность потока в мс
    tot_fwd_pkts = models.IntegerField(default=0, null=True, blank=True)  # Количество отправленных пакетов
    tot_bwd_pkts = models.IntegerField(default=0, null=True, blank=True)  # Количество полученных пакетов
    totlen_fwd_pkts = models.FloatField(default=0.0, null=True, blank=True)  # Общая длина отправленных пакетов
    totlen_bwd_pkts = models.FloatField(default=0.0, null=True, blank=True)  # Общая длина полученных пакетов
    fwd_pkt_len_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальная длина отправленного пакета
    fwd_pkt_len_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальная длина отправленного пакета
    fwd_pkt_len_mean = models.FloatField(default=0.0, null=True, blank=True)  # Средняя длина отправленных пакетов
    fwd_pkt_len_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение длины отправленных пакетов
    bwd_pkt_len_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальная длина полученного пакета
    bwd_pkt_len_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальная длина полученного пакета
    bwd_pkt_len_mean = models.FloatField(default=0.0, null=True, blank=True)  # Средняя длина полученных пакетов
    bwd_pkt_len_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение длины полученных пакетов
    flow_byts_per_sec = models.FloatField(default=0.0, null=True, blank=True)  # Скорость передачи байтов
    flow_pkts_per_sec = models.FloatField(default=0.0, null=True, blank=True)  # Скорость передачи пакетов
    flow_iat_mean = models.FloatField(default=0.0, null=True, blank=True)  # Средний интервал между потоками
    flow_iat_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение интервала между потоками
    flow_iat_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальный интервал между потоками
    flow_iat_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальный интервал между потоками
    fwd_iat_tot = models.FloatField(default=0.0, null=True, blank=True)  # Общий интервал отправленных пакетов
    fwd_iat_mean = models.FloatField(default=0.0, null=True, blank=True)  # Средний интервал отправленных пакетов
    fwd_iat_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение интервала отправленных пакетов
    fwd_iat_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальный интервал отправленных пакетов
    fwd_iat_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальный интервал отправленных пакетов
    bwd_iat_tot = models.FloatField(default=0.0, null=True, blank=True)  # Общий интервал полученных пакетов
    bwd_iat_mean = models.FloatField(default=0.0, null=True, blank=True)  # Средний интервал полученных пакетов
    bwd_iat_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение интервала полученных пакетов
    bwd_iat_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальный интервал полученных пакетов
    bwd_iat_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальный интервал полученных пакетов
    fwd_psh_flags = models.IntegerField(default=0, null=True, blank=True)  # Количество PSH-флагов в отправленных пакетах
    bwd_psh_flags = models.IntegerField(default=0, null=True, blank=True)  # Количество PSH-флагов в полученных пакетах
    fwd_urg_flags = models.IntegerField(default=0, null=True, blank=True)  # Количество URG-флагов в отправленных пакетах
    bwd_urg_flags = models.IntegerField(default=0, null=True, blank=True)  # Количество URG-флагов в полученных пакетах
    fwd_header_len = models.IntegerField(default=0, null=True, blank=True)  # Длина заголовка в отправленных пакетах
    bwd_header_len = models.IntegerField(default=0, null=True, blank=True)  # Длина заголовка в полученных пакетах
    fwd_pkts_per_sec = models.FloatField(default=0.0, null=True, blank=True)  # Скорость передачи отправленных пакетов
    bwd_pkts_per_sec = models.FloatField(default=0.0, null=True, blank=True)  # Скорость передачи полученных пакетов
    pkt_len_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальная длина пакета
    pkt_len_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальная длина пакета
    pkt_len_mean = models.FloatField(default=0.0, null=True, blank=True)  # Средняя длина пакета
    pkt_len_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение длины пакета
    pkt_len_var = models.FloatField(default=0.0, null=True, blank=True)  # Дисперсия длины пакета
    fin_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество FIN-флагов
    syn_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество SYN-флагов
    rst_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество RST-флагов
    psh_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество PSH-флагов
    ack_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество ACK-флагов
    urg_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество URG-флагов
    cwe_flag_count = models.IntegerField(default=0, null=True, blank=True)  # Количество CWE-флагов
    ece_flag_cnt = models.IntegerField(default=0, null=True, blank=True)  # Количество ECE-флагов
    down_up_ratio = models.FloatField(default=0.0, null=True, blank=True)  # Соотношение данных вниз/вверх
    pkt_size_avg = models.FloatField(default=0.0, null=True, blank=True)  # Средний размер пакета
    fwd_seg_size_avg = models.FloatField(default=0.0, null=True, blank=True)  # Средний размер сегмента отправленных пакетов
    bwd_seg_size_avg = models.FloatField(default=0.0, null=True, blank=True)  # Средний размер сегмента полученных пакетов
    fwd_byts_per_avg = models.FloatField(default=0.0, null=True, blank=True)  # Среднее количество байтов в отправленных пакетах
    fwd_pkts_per_avg = models.FloatField(default=0.0, null=True, blank=True)  # Среднее количество пакетов в отправленных пакетах
    fwd_blk_rate_avg = models.FloatField(default=0.0, null=True, blank=True)  # Средняя скорость блокировки в отправленных пакетах
    bwd_byts_per_avg = models.FloatField(default=0.0, null=True, blank=True)  # Среднее количество байтов в полученных пакетах
    bwd_pkts_per_avg = models.FloatField(default=0.0, null=True, blank=True)  # Среднее количество пакетов в полученных пакетах
    bwd_blk_rate_avg = models.FloatField(default=0.0, null=True, blank=True)  # Средняя скорость блокировки в полученных пакетах
    subflow_fwd_pkts = models.IntegerField(default=0, null=True, blank=True)  # Количество отправленных подпотоков
    subflow_fwd_byts = models.IntegerField(default=0, null=True, blank=True)  # Количество байтов в отправленных подпотоках
    subflow_bwd_pkts = models.IntegerField(default=0, null=True, blank=True)  # Количество полученных подпотоков
    subflow_bwd_byts = models.IntegerField(default=0, null=True, blank=True)  # Количество байтов в полученных подпотоках
    init_fwd_win_byts = models.IntegerField(default=0, null=True, blank=True)  # Начальный размер окна отправленных пакетов
    init_bwd_win_byts = models.IntegerField(default=0, null=True, blank=True)  # Начальный размер окна полученных пакетов
    fwd_act_data_pkts = models.IntegerField(default=0, null=True, blank=True)  # Количество активных данных в отправленных пакетах
    fwd_seg_size_min = models.IntegerField(default=0, null=True, blank=True)  # Минимальный размер сегмента в отправленных пакетах
    active_mean = models.FloatField(default=0.0, null=True, blank=True)  # Среднее время активности потока
    active_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение времени активности
    active_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальное время активности
    active_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальное время активности
    idle_mean = models.FloatField(default=0.0, null=True, blank=True)  # Среднее время простоя потока
    idle_std = models.FloatField(default=0.0, null=True, blank=True)  # Стандартное отклонение времени простоя
    idle_max = models.FloatField(default=0.0, null=True, blank=True)  # Максимальное время простоя
    idle_min = models.FloatField(default=0.0, null=True, blank=True)  # Минимальное время простоя
    label = models.CharField(max_length=50, default='', null=True, blank=True)  # Метка (пустая по умолчанию)

    def __str__(self):
        return f"ID {self.id} Flow {self.flow_id} from {self.src_ip} to {self.dst_ip}"