#firstAI/models.py

from django.db import models

class Weight(models.Model):
    ATTRIBUTE_CHOICES = [
        ('src_ip', 'Source IP'),
        ('dst_ip', 'Destination IP'),
        ('src_port', 'Source Port'),
        ('dst_port', 'Destination Port'),
        ('timestamp', 'Timestamp'),
        ('flow_duration', 'Flow Duration'),
        ('tot_fwd_pkts', 'Total Forward Packets'),
        ('tot_bwd_pkts', 'Total Backward Packets'),
        ('flow_byts_per_sec', 'Flow Bytes Per Second'),
        ('flow_pkts_per_sec', 'Flow Packets Per Second'),
        ('fwd_iat_mean', 'Forward IAT Mean'),
        ('bwd_iat_mean', 'Backward IAT Mean'),
        ('protocol', 'Protocol'),
        ('down_up_ratio', 'Down-Up Ratio'),
        ('pkt_size_avg', 'Packet Size Average'),
        ('fwd_pkts_per_sec', 'Forward Packets Per Second'),
        ('bwd_pkts_per_sec', 'Backward Packets Per Second')
    ]

    attribute = models.CharField(max_length=50, choices=ATTRIBUTE_CHOICES, unique=True)
    weight = models.FloatField(default=0.0)
    is_active = models.BooleanField(default=True)  # Поле для отметки активности

    def __str__(self):
        return f"{self.get_attribute_display()}: {self.weight} (Активен: {self.is_active})"
