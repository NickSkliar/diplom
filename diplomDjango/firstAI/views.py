# views.py

from django.shortcuts import render, redirect
from database.models import DownloadDB
from .models import Weight
import json
import os

def table_view(request):
    # Existing code to handle 'start' and 'end' parameters
    start = request.GET.get('start')
    end = request.GET.get('end')

    try:
        start = int(start)
    except (TypeError, ValueError):
        start = 0  # Default value

    try:
        end = int(end)
    except (TypeError, ValueError):
        end = 20  # Default value

    if start < 0:
        start = 0
    if end <= start:
        end = start + 20

    # Load data from the database
    datas = DownloadDB.objects.values(
        'id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'flow_duration',
        'tot_fwd_pkts', 'tot_bwd_pkts', 'flow_byts_per_sec', 'flow_pkts_per_sec',
        'fwd_iat_mean', 'bwd_iat_mean', 'protocol', 'down_up_ratio',
        'pkt_size_avg', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'label'
    )[start:end]

    weights = Weight.objects.all()

    # New code to read JSON files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Read 'rest_metricks.json'
    rest_metrics_file = os.path.join(base_dir, 'results/test_metrics.json')
    with open(rest_metrics_file, 'r') as f:
        rest_metrics = json.load(f)

    # Read 'training_results.json'
    training_results_file = os.path.join(base_dir, 'models/training_results.json')
    with open(training_results_file, 'r') as f:
        training_results = json.load(f)

    # Pass the data to the template
    return render(request, 'firstAI/AI1_page.html', {
        'datas': datas,
        'start': start,
        'end': end,
        'weights': weights,
        'rest_metrics': rest_metrics,
        'training_results': training_results,
    })
