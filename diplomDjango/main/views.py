from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def main_page(request):
    return render(request, 'main/main_page.html')

def AI1_page(request):
    return render(request, 'firstAI/AI1_page.html')
