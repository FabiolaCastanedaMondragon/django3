import json
import numpy as np
import pandas as pd
from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ======================================================================
# SIMULACIÓN DE DATOS Y ENTRENAMIENTO DE MODELO
# ======================================================================

def generate_simulated_data(n_samples=500):
    """
    Genera datos simulados, incluyendo las nuevas características
    para que la tabla sea más grande (como en la imagen del usuario).
    """
    np.random.seed(42)
    
    # 1. Generación de características base
    duration = np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples)
    total_fpackets = np.random.poisson(lam=15, size=n_samples)
    
    # 2. Generar etiquetas (0: Benigno, 1: Malware)
    # AJUSTE CLAVE para aumentar la probabilidad de Malware (puntos rojos)
    malware_prob = 1 / (1 + np.exp(-(0.5 * duration + 0.1 * total_fpackets - 5)))
    labels = (np.random.rand(n_samples) < malware_prob).astype(int)
    
    data = pd.DataFrame({
        'duration': duration.round(2),
        'total_fpackets': total_fpackets,
        'activity_count': np.random.randint(1, 50, n_samples),
        'class': labels
    })
    
    # 3. Ajuste para mejor separabilidad
    malware_filter = data['class'] == 1
    # Aumentar los paquetes y actividad en casos de malware
    data.loc[malware_filter, 'total_fpackets'] = data.loc[malware_filter, 'total_fpackets'] + np.random.randint(5, 15, size=(malware_filter.sum()))
    data.loc[malware_filter, 'activity_count'] = data.loc[malware_filter, 'activity_count'] + np.random.randint(10, 30, size=(malware_filter.sum()))
    
    # Reducir la duración en casos benignos
    data.loc[~malware_filter, 'duration'] = data.loc[~malware_filter, 'duration'] - np.random.uniform(0, 1, size=( (~malware_filter).sum() ))
    data['duration'] = np.clip(data['duration'], 0.1, None)

    # 4. SIMULACIÓN DE CARACTERÍSTICAS ADICIONALES (Para llenar la tabla)
    # Generaremos 10 columnas adicionales que se ven en la imagen.
    data['TOTAL BYTES'] = data['total_fpackets'] * np.random.randint(500, 2000, n_samples)
    data['TOTAL RPKTS'] = data['total_fpackets'] * np.random.uniform(0.5, 0.9, n_samples)
    data['TOTAL RBYTES'] = data['TOTAL BYTES'] * np.random.uniform(0.3, 0.7, n_samples)
    
    # Generación de métricas estadísticas de paquetes/bytes
    base_pkts = data['total_fpackets'] * (data['class'] + 1) * 0.5 # Aumenta en malware
    base_bytes = data['TOTAL BYTES'] / 100000 

    data['MIN PKTS'] = np.clip(base_pkts * np.random.uniform(0.05, 0.2), 1, None).round(2)
    data['MAX PKTS'] = np.clip(base_pkts * np.random.uniform(1.5, 3.0), 5, None).round(2)
    data['MEAN PKTS'] = (data['MIN PKTS'] + data['MAX PKTS']) / 2
    data['MEAN BYTES'] = np.clip(base_bytes * np.random.uniform(0.8, 1.5), 10, None).round(2)
    data['STDEV BYTES'] = data['MEAN BYTES'] * np.random.uniform(0.2, 0.5, n_samples)
    data['MIN BYTES'] = np.clip(data['MEAN BYTES'] - data['STDEV BYTES'], 10, None).round(2)
    data['MAX BYTES'] = np.clip(data['MEAN BYTES'] + data['STDEV BYTES'] * 2, 50, None).round(2)

    # 5. Normalización para el modelo y la gráfica
    # Usaremos solo 'duration' y 'total_fpackets' para el entrenamiento por simplicidad
    data['Duration_Scaled'] = (data['duration'] - data['duration'].min()) / (data['duration'].max() - data['duration'].min())
    data['TotalFpackets_Scaled'] = (data['total_fpackets'] - data['total_fpackets'].min()) / (data['total_fpackets'].max() - data['total_fpackets'].min())

    return data

def train_simulated_model(df):
    """Entrena un modelo Random Forest y calcula la precisión y F1 Score."""
    
    # Usamos las características escaladas para el modelo
    X = df[['Duration_Scaled', 'TotalFpackets_Scaled']]
    y = df['class']

    if len(X) < 2:
        return None, 0.0, 0.0, pd.DataFrame(), pd.Series()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # El cálculo real de precisión y F1 (que ignoraremos en favor del valor fijo)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # === AJUSTE DE VALOR FIJO SOLICITADO ===
    # El usuario pide que el F1 Score sea 0.80
    f1 = 0.8000 
    
    return model, accuracy, f1, X_test, y_test

# --- INICIALIZACIÓN GLOBAL ---
try:
    SIMULATED_DATA = generate_simulated_data(n_samples=500)
    # Solo necesitamos ACCURACY para la inicialización, pero su valor será ignorado
    MODEL, ACCURACY, F1_SCORE, X_TEST, Y_TEST = train_simulated_model(SIMULATED_DATA.copy())
    INITIALIZATION_SUCCESS = True
except Exception as e:
    print(f"Error al inicializar el modelo de simulación: {e}")
    SIMULATED_DATA = pd.DataFrame()
    MODEL = None
    ACCURACY = 0.0
    F1_SCORE = 0.0
    X_TEST = pd.DataFrame()
    Y_TEST = pd.Series()
    INITIALIZATION_SUCCESS = False


# ======================================================================
# VISTA DEL API DE DJANGO (malware_results)
# ======================================================================

def malware_results(request):
    """
    Endpoint del API que devuelve los resultados del análisis del modelo
    de detección de malware.
    """
    
    if not INITIALIZATION_SUCCESS or MODEL is None or SIMULATED_DATA.empty:
        error_msg = "Error: El modelo de simulación no pudo ser inicializado. Asegúrate de que numpy, pandas y scikit-learn estén instalados."
        return JsonResponse({'error': error_msg}, status=500)

    try:
        # 1. Preparar los datos del DataFrame para la tabla (100 primeras filas)
        df_for_display = SIMULATED_DATA.head(100).copy() 
        
        # Generar predicciones para las filas mostradas (usa las características escaladas)
        features_sample = df_for_display[['Duration_Scaled', 'TotalFpackets_Scaled']]
        df_for_display['prediction'] = MODEL.predict(features_sample)
        
        # Renombrar columnas para visualización en español
        df_for_display.rename(columns={
            'duration': 'DURACION (seg)',
            'total_fpackets': 'TOTAL FPKTS',
            'activity_count': 'ACT. COUNT',
            'class': 'CLASE',
            'prediction': 'PREDICCION'
        }, inplace=True)
        
        # Definir las columnas a mostrar, manteniendo el orden de la tabla simulada
        display_cols = [
            'DURACION (seg)', 'TOTAL FPKTS', 'TOTAL BYTES', 'TOTAL RPKTS', 
            'TOTAL RBYTES', 'MIN PKTS', 'MAX PKTS', 'MEAN PKTS', 
            'MEAN BYTES', 'STDEV BYTES', 'MIN BYTES', 'MAX BYTES',
            'ACT. COUNT', 'CLASE', 'PREDICCION'
        ]
        
        # Eliminar las columnas escaladas del display
        df_for_display = df_for_display[[col for col in display_cols if col in df_for_display.columns]]
        
        # Convertir a formato de lista de diccionarios para JSON
        dataframe_json = df_for_display.to_dict('records')

        # 2. Preparar los datos de separabilidad para el gráfico de dispersión
        separability_data = {
            'x_points': X_TEST['Duration_Scaled'].tolist(),
            'y_points': X_TEST['TotalFpackets_Scaled'].tolist(),
            'labels': Y_TEST.tolist() # 0 o 1
        }
        
        # 3. Construir la respuesta JSON final
        response_data = {
            'metrics': {
                # Se elimina la precisión (accuracy) del JSON
                'f1_score': F1_SCORE # Este valor ahora es fijo en 0.8000
            },
            'separability_data': separability_data,
            'dataframe_sample': dataframe_json
        }
        
        return JsonResponse(response_data)

    except Exception as e:
        error_message = f"Error interno del servidor al procesar datos: {str(e)}"
        print(error_message)
        return JsonResponse({'error': error_message}, status=500)
