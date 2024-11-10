from django.db import models


class download_db(models.Model):
    custom_id = models.AutoField(primary_key=True)
    flow_id = models.CharField(max_length=255)  #треба для того, щоб ідентифікувати кожен потік даних унікально
    src_ip = models.GenericIPAddressField() #треба для вказівки IP-адреси джерела, звідки розпочато передачу даних
    src_port = models.IntegerField()  #використовується для збереження порту джерела, через який передаються дані
    dst_ip = models.GenericIPAddressField() #треба для фіксації IP-адреси призначення, куди спрямований потік даних
    dst_port = models.IntegerField() #зберігає порт призначення для прийому потоку даних
    protocol = models.IntegerField() #вказує протокол передачі (наприклад, TCP, UDP)
    #timestamp = models.DateTimeField() #зберігає тимчасову позначку початку передачі даних
    flow_duration = models.BigIntegerField() #показує тривалість передачі потоку в мілісекундах
    tot_fwd_pkts = models.IntegerField()  #використовується для обліку загальної кількості пакетів, відправлених вперед
    tot_bwd_pkts = models.IntegerField()  #показує загальну кількість пакетів, відправлених назад
    totlen_fwd_pkts = models.FloatField()  #треба для того, щоб відстежувати загальну довжину пакетів, відправлених вперед
    totlen_bwd_pkts = models.FloatField()  # використовується для вимірювання загальної довжини пакетів, відправлених назад
    fwd_pkt_len_max = models.FloatField()  # зберігає максимальну довжину пакета, відправленого вперед
    fwd_pkt_len_min = models.FloatField()  # показує мінімальну довжину пакета, відправленого вперед
    fwd_pkt_len_mean = models.FloatField()  # використовується для обчислення середньої довжини пакетів, відправлених вперед
    fwd_pkt_len_std = models.FloatField()  # показує стандартне відхилення довжини відправлених пакетів
    bwd_pkt_len_max = models.FloatField()  # фіксує максимальну довжину пакета, що йде назад
    bwd_pkt_len_min = models.FloatField()  # зберігає мінімальну довжину пакета, що йде назад
    bwd_pkt_len_mean = models.FloatField()  # треба для обчислення середньої довжини пакетів, що йдуть назад
    bwd_pkt_len_std = models.FloatField()  # використовується для визначення стандартного відхилення довжини пакетів, що йдуть назад
    flow_byts_per_sec = models.FloatField()  # показує швидкість передачі байтів за секунду
    flow_pkts_per_sec = models.FloatField()  # вказує швидкість передачі пакетів за секунду
    flow_iat_mean = models.FloatField()  # зберігає середній інтервал між потоками
    flow_iat_std = models.FloatField()  # показує стандартне відхилення інтервалу між потоками
    flow_iat_max = models.FloatField()  # фіксує максимальний інтервал між потоками
    flow_iat_min = models.FloatField()  # вказує мінімальний інтервал між потоками
    fwd_iat_tot = models.FloatField()  # зберігає загальний інтервал між відправленими пакетами
    fwd_iat_mean = models.FloatField()  # використовується для обчислення середнього інтервалу між відправленими пакетами
    fwd_iat_std = models.FloatField()  # показує стандартне відхилення інтервалу між відправленими пакетами
    fwd_iat_max = models.FloatField()  # фіксує максимальний інтервал між відправленими пакетами
    fwd_iat_min = models.FloatField()  # показує мінімальний інтервал між відправленими пакетами
    bwd_iat_tot = models.FloatField()  # зберігає загальний інтервал між пакетами, що йдуть назад
    bwd_iat_mean = models.FloatField()  # використовується для обчислення середнього інтервалу між пакетами, що йдуть назад
    bwd_iat_std = models.FloatField()  # показує стандартне відхилення інтервалу між пакетами, що йдуть назад
    bwd_iat_max = models.FloatField()  # фіксує максимальний інтервал між пакетами, що йдуть назад
    bwd_iat_min = models.FloatField()  # вказує мінімальний інтервал між пакетами, що йдуть назад
    fwd_psh_flags = models.IntegerField()  # використовується для обліку прапорців PSH у відправлених пакетах
    bwd_psh_flags = models.IntegerField()  # показує кількість прапорців PSH у пакетах, що йдуть назад
    fwd_urg_flags = models.IntegerField()  # зберігає кількість прапорців URG у відправлених пакетах
    bwd_urg_flags = models.IntegerField()  # показує кількість прапорців URG у пакетах, що йдуть назад
    fwd_header_len = models.IntegerField()  # вказує довжину заголовку у відправлених пакетах
    bwd_header_len = models.IntegerField()  # вказує довжину заголовку у пакетах, що йдуть назад
    fwd_pkts_per_sec = models.FloatField()  # показує швидкість передачі відправлених пакетів за секунду
    bwd_pkts_per_sec = models.FloatField()  # показує швидкість передачі пакетів, що йдуть назад, за секунду
    pkt_len_min = models.FloatField()  # вказує мінімальну довжину пакета
    pkt_len_max = models.FloatField()  # фіксує максимальну довжину пакета
    pkt_len_mean = models.FloatField()  # використовується для обчислення середньої довжини пакета
    pkt_len_std = models.FloatField()  # показує стандартне відхилення довжини пакета
    pkt_len_var = models.FloatField()  # зберігає дисперсію довжини пакета
    fin_flag_cnt = models.IntegerField()  # вказує кількість прапорців FIN
    syn_flag_cnt = models.IntegerField()  # зберігає кількість прапорців SYN
    rst_flag_cnt = models.IntegerField()  # фіксує кількість прапорців RST
    psh_flag_cnt = models.IntegerField()  # зберігає кількість прапорців PSH
    ack_flag_cnt = models.IntegerField()  # показує кількість прапорців ACK
    urg_flag_cnt = models.IntegerField()  # використовується для обліку прапорців URG
    cwe_flag_count = models.IntegerField()  # зберігає кількість прапорців CWE
    ece_flag_cnt = models.IntegerField()  # вказує кількість прапорців ECE
    down_up_ratio = models.FloatField()  # показує співвідношення переданих даних вниз/вгору
    pkt_size_avg = models.FloatField()  # використовується для обчислення середнього розміру пакета
    fwd_seg_size_avg = models.FloatField()  # зберігає середній розмір сегмента у відправлених пакетах
    bwd_seg_size_avg = models.FloatField()  # показує середній розмір сегмента у пакетах, що йдуть назад
    fwd_byts_per_avg = models.FloatField()  # вказує середню кількість байтів у відправлених пакетах
    fwd_pkts_per_avg = models.FloatField()  # використовується для обчислення середньої кількості пакетів у відправлених пакетах
    fwd_blk_rate_avg = models.FloatField()  # показує середню швидкість блокування у відправлених пакетах
    bwd_byts_per_avg = models.FloatField()  # зберігає середню кількість байтів у пакетах, що йдуть назад
    bwd_pkts_per_avg = models.FloatField()  # вказує середню кількість пакетів у пакетах, що йдуть назад
    bwd_blk_rate_avg = models.FloatField()  # показує середню швидкість блокування у пакетах, що йдуть назад
    subflow_fwd_pkts = models.IntegerField()  # використовується для обліку кількості підпотоків, відправлених вперед
    subflow_fwd_byts = models.IntegerField()  # зберігає кількість байтів підпотоків, відправлених вперед
    subflow_bwd_pkts = models.IntegerField()  # фіксує кількість підпотоків, що йдуть назад
    subflow_bwd_byts = models.IntegerField()  # вказує кількість байтів підпотоків, що йдуть назад
    init_fwd_win_byts = models.IntegerField()  # зберігає початковий розмір вікна для відправлених пакетів
    init_bwd_win_byts = models.IntegerField()  # вказує початковий розмір вікна для пакетів, що йдуть назад
    fwd_act_data_pkts = models.IntegerField()  # показує кількість активних пакетів даних у відправлених пакетах
    fwd_seg_size_min = models.IntegerField()  # вказує мінімальний розмір сегмента у відправлених пакетах
    active_mean = models.FloatField()  # використовується для обчислення середнього часу активності потоку
    active_std = models.FloatField()  # показує стандартне відхилення активності потоку
    active_max = models.FloatField()  # фіксує максимальну тривалість активності потоку
    active_min = models.FloatField()  # вказує мінімальну тривалість активності потоку
    idle_mean = models.FloatField()  # використовується для обчислення середнього часу бездіяльності потоку
    idle_std = models.FloatField()  # показує стандартне відхилення бездіяльності потоку
    idle_max = models.FloatField()  # фіксує максимальний час бездіяльності потоку
    idle_min = models.FloatField()  # вказує мінімальний час бездіяльності потоку
    label = models.CharField(max_length=50)  # використовується для збереження мітки або категорії потоку, наприклад, для класифікації (наприклад, «ddos»)
    def __str__(self):
        return f"Flow {self.flow_id} from {self.src_ip} to {self.dst_ip}"

