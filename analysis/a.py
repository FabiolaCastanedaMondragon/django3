import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import json 

def run_malware_analysis():
    """
    Carga datos, entrena un modelo Random Forest,
    y devuelve las métricas (Accuracy, F1 Score), una muestra del dataset
    y los datos ESCALADOS para la gráfica de separabilidad.
    """
    # --- 1. Cargar y Limpiar Datos ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'TotalFeatures-ISCXFlowMeter.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}. Asegúrate de que está en la ruta correcta.")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Limpieza: reemplazar infinitos y NaN con 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    target = 'calss'
    features = ['duration', 'total_fpackets', 'total_bpktl']

    if target not in df.columns:
        raise ValueError(f"Columna objetivo '{target}' no encontrada.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # Factorizar la columna objetivo a números enteros (0, 1, 2, ...)
    y = df[target].factorize()[0]
    X = df[features]

    # --- 2. División y Escalado ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Extraer las columnas de prueba escaladas para la gráfica
    idx_duration = features.index('duration')
    idx_fpackets = features.index('total_fpackets')
    X_test_duration_scaled = X_test_scaled[:, idx_duration]
    X_test_fpackets_scaled = X_test_scaled[:, idx_fpackets]

    # --- 3. Entrenar Modelo y Métricas ---
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Cálculo de Métricas
    accuracy = accuracy_score(y_test, y_pred)
    # CORRECCIÓN: Usar 'weighted' porque los datos son multiclase, no binarios.
    f1 = f1_score(y_test, y_pred, average='weighted') 

    # --- 4. Crear Malla para Frontera de Decisión ---
    
    # Rangos de los datos de prueba ESCALADOS
    x_min_scaled, x_max_scaled = X_test_duration_scaled.min(), X_test_duration_scaled.max()
    y_min_scaled, y_max_scaled = X_test_fpackets_scaled.min(), X_test_fpackets_scaled.max()
    
    # Crear malla (Grid) con valores escalados
    xx_scaled, yy_scaled = np.meshgrid(np.linspace(x_min_scaled, x_max_scaled, 100),
                                       np.linspace(y_min_scaled, y_max_scaled, 100))

    # Reconstruir el grid de 3 características escaladas
    grid_scaled_array = np.zeros((xx_scaled.size, len(features)))
    grid_scaled_array[:, idx_duration] = xx_scaled.ravel()
    grid_scaled_array[:, idx_fpackets] = yy_scaled.ravel()
    
    # Rellenar la tercera característica ('total_bpktl') con su valor medio ESCALADO
    mean_scaled_bpktl = X_train_scaled[:, features.index('total_bpktl')].mean()
    grid_scaled_array[:, features.index('total_bpktl')] = mean_scaled_bpktl

    # Predecir la probabilidad de ser la primera clase (0) - Esto es general para el gráfico de contorno
    # Usaremos la primera columna de predict_proba para simplificar
    zz = model.predict_proba(grid_scaled_array)[:, 0].reshape(xx_scaled.shape) 
    
    # --- 5. Salida Estructurada ---
    df_sample = df.head(10).to_dict('records')

    results_dict = {
        'metrics': {
            'accuracy': round(accuracy, 8),
            'f1_score': round(f1, 8) 
        },
        'dataframe_sample': df_sample,
        'separability_data': {
            # Datos de la malla para la frontera de decisión
            'xx': xx_scaled.tolist(),
            'yy': yy_scaled.tolist(),
            'zz': zz.tolist(),
            # Puntos de prueba escalados y sus etiquetas
            'x_points': X_test_duration_scaled.tolist(),
            'y_points': X_test_fpackets_scaled.tolist(),
            'labels': y_test.tolist()
        }
    }
    
    return results_dict

def generate_json_output(filename='malware_analysis_data.json'):
    """
    Ejecuta el análisis completo y guarda el resultado en un archivo JSON.
    """
    print("Iniciando análisis y entrenamiento del modelo...")
    try:
        data = run_malware_analysis()
    except Exception as e:
        print(f"Error al ejecutar el análisis: {e}")
        return

    # Guarda el diccionario en un archivo JSON
    # Se añade sort_keys=True para garantizar un orden consistente si es necesario
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\n¡Análisis completado! 🎉\nLos datos se han guardado en: {filename}")

# --- Ejecución Principal ---
if __name__ == '__main__':
    generate_json_output()