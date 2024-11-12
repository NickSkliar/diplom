from django.shortcuts import render, redirect
from database.models import download_db
from .models import Weight


def table_view(request):
    # Получаем значения start и end из GET-параметров запроса
    start = request.GET.get('start')
    end = request.GET.get('end')

    # Проверяем, что start и end заданы и корректны
    if start and end and start.isdigit() and end.isdigit():
        start = int(start)
        end = int(end)

        # Убедимся, что start меньше end
        if start >= 0 and end > start:
            # Загружаем данные только в указанном диапазоне
            datas = download_db.objects.values(
                'id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'flow_duration',
                'tot_fwd_pkts', 'tot_bwd_pkts', 'flow_byts_per_sec', 'flow_pkts_per_sec',
                'fwd_iat_mean', 'bwd_iat_mean', 'protocol', 'down_up_ratio',
                'pkt_size_avg', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'label'
            )[start:end]
        else:
            datas = []  # Пустой список, если значения некорректны
    else:
        datas = []  # Пустой список до выполнения запроса


    weights = Weight.objects.all()


    # Передаем данные и значения start и end в шаблон
    return render(request, 'firstAI/AI1_page.html', {
        'datas': datas,
        'start': start or '0',
        'end': end or '20',
        'weights': weights,
    })


