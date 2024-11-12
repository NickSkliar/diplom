from django.contrib import admin
from .models import Weight

@admin.register(Weight)
class FeatureAdmin(admin.ModelAdmin):
    list_display = ('attribute', 'weight', 'is_active')
    list_editable = ('weight', 'is_active')
