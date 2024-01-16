import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Crear un DataFrame desde el archivo CSV
df = pd.read_csv('DATAVH.csv')

# Seleccionar un producto específico con el código 
codigo_producto_seleccionado = '3260300'
df_producto = df[df['codigo'] == codigo_producto_seleccionado]

# Convertir la columna 'fecha' a formato datetime
df_producto['fecha'] = pd.to_datetime(df_producto['fecha'], dayfirst=True)

# Convertir columnas categóricas a numéricas usando Label Encoding
label_encoder = LabelEncoder()
df_producto['vendedor'] = label_encoder.fit_transform(df_producto['vendedor'])
df_producto['area'] = label_encoder.fit_transform(df_producto['area'])
df_producto['actividad_economica'] = label_encoder.fit_transform(df_producto['actividad_economica'])

X_producto = df_producto[['cantidad', 'vendedor', 'area', 'actividad_economica']]
y_producto = df_producto['cantidad']

# Asegurar que haya suficientes muestras para dividir
if len(df_producto) > 1:
    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train_producto, X_test_producto, y_train_producto, y_test_producto = train_test_split(X_producto, y_producto, test_size=0.2, random_state=0)

    # Inicializar el modelo de regresión lineal
    model_producto = LinearRegression()

    # Entrenar el modelo con los datos de entrenamiento
    model_producto.fit(X_train_producto, y_train_producto)

    # Realizar predicciones en el conjunto de prueba
    y_pred_producto = model_producto.predict(X_test_producto)

    # Evaluar el rendimiento del modelo
    print('Coeficientes:', model_producto.coef_)
    print('Intercepto:', model_producto.intercept_)
    print('Error absoluto medio:', metrics.mean_absolute_error(y_test_producto, y_pred_producto))
    print('Error cuadrático medio:', metrics.mean_squared_error(y_test_producto, y_pred_producto))
    print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test_producto, y_pred_producto)))

    # Graficar la regresión lineal
    plt.scatter(X_test_producto['cantidad'], y_test_producto, color='black')
    plt.plot(X_test_producto['cantidad'], y_pred_producto, color='blue', linewidth=3)
    plt.title('Regresión Lineal - Proyección de Ventas por Producto')
    plt.xlabel('Cantidad')
    plt.ylabel('Ventas')
    plt.show()

    # Graficar la diferencia entre los valores reales y predichos
    plt.scatter(y_test_producto, y_pred_producto)
    plt.title('Diferencia entre Valores Reales y Predichos')
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.show()

    # Graficar el error absoluto
    plt.scatter(X_test_producto['cantidad'], abs(y_test_producto - y_pred_producto))
    plt.title('Error Absoluto')
    plt.xlabel('Cantidad')
    plt.ylabel('Error Absoluto')
    plt.show()

else:
    print("No hay suficientes muestras para dividir en conjuntos de entrenamiento y prueba.")
