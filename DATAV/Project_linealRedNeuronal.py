import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Crear un DataFrame desde el archivo CSV
df = pd.read_csv('DATAV/DATAVHL.csv')

# Convertir la columna 'fecha' a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)

# Convertir columnas categóricas a numéricas usando Label Encoding
label_encoder = LabelEncoder()
df['vendedor'] = label_encoder.fit_transform(df['vendedor'])
df['area'] = label_encoder.fit_transform(df['area'])
df['actividad_economica'] = label_encoder.fit_transform(df['actividad_economica'])

# Seleccionar características (X) y variable objetivo (y)
X = df[['vendedor', 'area', 'actividad_economica', 'cantidad']]  # Incluir la columna 'cantidad' en X
y = df['cantidad']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos para la red neuronal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar el modelo de red neuronal
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Capa de salida para regresión lineal
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

# Realizar predicciones en el conjunto de prueba
y_pred_nn = model.predict(X_test_scaled).flatten()

# Evaluar el rendimiento del modelo
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)

print('Error absoluto medio (Red Neuronal):', mae_nn)
print('Error cuadrático medio (Red Neuronal):', mse_nn)
print('Raíz del error cuadrático medio (Red Neuronal):', rmse_nn)

# Graficar la regresión lineal y la red neuronal
plt.scatter(X_test['cantidad'], y_test, color='black', label='Datos reales')
plt.scatter(X_test['cantidad'], y_pred_nn, color='red', label='Red Neuronal')
plt.title('Comparación entre Regresión Lineal y Red Neuronal')
plt.xlabel('Cantidad')
plt.ylabel('Ventas')
plt.legend()
plt.show()

# Graficar el historial de entrenamiento
plt.plot(history.history['loss'], label='Perdida del entrenamiento')
plt.plot(history.history['val_loss'], label='Perdidad de validación')
plt.title('Historial de Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
