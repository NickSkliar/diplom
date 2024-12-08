from django.db import models
import uuid  # Необходимо для генерации UUID

class DownloadDB(models.Model):
    DATA_TYPE_CHOICES = [
        ('train', 'Training'),
        ('test', 'Testing'),
        ('unlabeled', 'Unlabeled'),
    ]

    id = models.AutoField(primary_key=True)
    src_ip = models.GenericIPAddressField()
    dst_ip = models.GenericIPAddressField()
    src_port = models.PositiveIntegerField()
    dst_port = models.PositiveIntegerField()
    protocol = models.PositiveSmallIntegerField()
    timestamp = models.DateTimeField(db_index=True)
    flow_duration = models.BigIntegerField()
    tot_fwd_pkts = models.PositiveIntegerField()
    tot_bwd_pkts = models.PositiveIntegerField()
    flow_byts_per_sec = models.FloatField()
    flow_pkts_per_sec = models.FloatField()
    fwd_iat_mean = models.FloatField()
    bwd_iat_mean = models.FloatField()
    down_up_ratio = models.FloatField()
    pkt_size_avg = models.FloatField()
    fwd_pkts_per_sec = models.FloatField()
    bwd_pkts_per_sec = models.FloatField()
    label = models.CharField(max_length=50, db_index=True)
    data_type = models.CharField(
        max_length=10,
        choices=DATA_TYPE_CHOICES,
        default='unlabeled',
        db_index=True
    )

    class Meta:
        db_table = "download_db"
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['label']),
            models.Index(fields=['data_type']),
            models.Index(fields=['src_ip']),
            models.Index(fields=['dst_ip']),
        ]
