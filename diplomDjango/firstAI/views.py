from django.shortcuts import render, redirect
from database.models import DownloadDB
from .models import Weight

def table_view(request):
    # Получаем значения start и end из GET-параметров запроса
    start = request.GET.get('start')
    end = request.GET.get('end')

    # Устанавливаем значения по умолчанию, если параметры отсутствуют или некорректны
    try:
        start = int(start)
    except (TypeError, ValueError):
        start = 0  # Значение по умолчанию

    try:
        end = int(end)
    except (TypeError, ValueError):
        end = 20  # Значение по умолчанию

    # Проверяем, что start и end имеют корректные значения
    if start < 0:
        start = 0
    if end <= start:
        end = start + 20  # Или любое другое логичное значение

    # Загружаем данные только в указанном диапазоне
    datas = DownloadDB.objects.values(
        'id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_duration',
        'tot_fwd_pkts', 'tot_bwd_pkts', 'flow_byts_per_sec', 'flow_pkts_per_sec',
        'fwd_iat_mean', 'bwd_iat_mean', 'protocol', 'down_up_ratio',
        'pkt_size_avg', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'label'
    )[start:end]

    weights = Weight.objects.all()

    # Передаем данные и значения start и end в шаблон
    return render(request, 'firstAI/AI1_page.html', {
        'datas': datas,
        'start': start,
        'end': end,
        'weights': weights,
    })


