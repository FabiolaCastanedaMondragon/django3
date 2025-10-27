from django.urls import path
from . import views

# Define el namespace de la aplicación (opcional pero recomendado)
app_name = 'analysis'

urlpatterns = [
    # Ruta: /api/malware/ (Llamará a la función views.malware_results)
    path('', views.malware_results, name='malware_api'),
]
