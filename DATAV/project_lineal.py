import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Crear un DataFrame desde el archivo CSV
df = pd.read_csv('DATAV/DATAVHL.csv')

# Convertir la columna 'fecha' a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)

# Convertir columnas categóricas a numéricas usando Label Encoding
label_encoder = LabelEncoder()
df['vendedor'] = label_encoder.fit_transform(df['vendedor'])
df['area'] = label_encoder.fit_transform(df['area'])
df['actividad_economica'] = label_encoder.fit_transform(df['actividad_economica'])

X = df[['cantidad', 'vendedor', 'area', 'actividad_economica']]
y = df['cantidad']  

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Inicializar el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
print('Coeficientes:', model.coef_)  # Coeficientes del modelo
print('Intercepto:', model.intercept_)  # Término de intercepción
print('Error absoluto medio:', metrics.mean_absolute_error(y_test, y_pred))
print('Error cuadrático medio:', metrics.mean_squared_error(y_test, y_pred))
print('Raíz del error cuadrático medio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Graficar la regresión lineal
plt.scatter(X_test['cantidad'], y_test, color='black')
plt.plot(X_test['cantidad'], y_pred, color='blue', linewidth=3)
plt.title('Regresión Lineal')
plt.xlabel('Cantidad')
plt.ylabel('Ventas')
plt.show()

# Graficar la diferencia entre los valores reales y predichos
plt.scatter(y_test, y_pred)
plt.title('Diferencia entre Valores Reales y Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.show()

# Graficar el error absoluto
plt.scatter(X_test['cantidad'], abs(y_test - y_pred))
plt.title('Error Absoluto')
plt.xlabel('Cantidad')
plt.ylabel('Error Absoluto')
plt.show()
