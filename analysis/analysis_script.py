import os
import json
from django.http import JsonResponse, HttpResponseServerError, HttpResponseNotFound

# Define la ruta absoluta al archivo JSON
# Esto garantiza que siempre encuentra el JSON dentro de la misma carpeta 'analysis/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'malware_analysis_data.json')

# Función que sirve los datos JSON
def api_malware_data(request): 
    """
    Lee el archivo JSON de análisis y lo sirve como respuesta JSON pura al frontend (HTML/JavaScript).
    """
    try:
        # 1. Leer el archivo JSON
        # Esta es la parte que "abre" el JSON que generaste con el script
        with open(DATA_FILE_PATH, 'r') as f:
            analysis_data = json.load(f)
        
        # 2. Devolver el JSON directamente al cliente (navegador)
        return JsonResponse(analysis_data, status=200)

    except FileNotFoundError:
        # Error 404 si el archivo JSON no existe (hay que ejecutar el script primero)
        print(f"ERROR 404: JSON de análisis no encontrado en: {DATA_FILE_PATH}")
        return HttpResponseNotFound("Error 404: Archivo de datos JSON no encontrado. Ejecuta el script de análisis primero.")
        
    except json.JSONDecodeError:
        # Error 500 si el JSON no se puede parsear
        print("ERROR 500: El archivo JSON está malformado.")
        return HttpResponseServerError("Error 500: El archivo JSON está malformado. Revisa el contenido.")
