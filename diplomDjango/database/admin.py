from django.contrib import admin
from .models import DownloadDB

@admin.register(DownloadDB)
class DownloadDataAdmin(admin.ModelAdmin):
    list_display = ('id',  'src_ip', 'dst_ip', 'timestamp', 'label', 'data_type')
    list_per_page = 50

