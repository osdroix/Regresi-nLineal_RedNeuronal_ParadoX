from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route('/home')
@app.route("/")
def home():
    return render_template("form.html", preview_data=None)

@app.route('/regresar')
def regresar_a_formulario():
    # Redirige a la ruta del formulario usando la función url_for
    return render_template("form.html", preview_data=None)

def calcular_porcentaje_top_tres(df, total_ingreso):
    top_tres_ingreso = df['ingreso_bruto'].sum()
    porcentaje = (top_tres_ingreso / total_ingreso) * 100
    return round(porcentaje, 3)

def realizar_regresion_lineal(df):
    # Convertir fechas a valores numéricos (días desde el inicio)
    df['fecha_numerica'] = (pd.to_datetime(df['fecha']) - pd.to_datetime(df['fecha']).min()).dt.days

    X = df[['fecha_numerica']]
    y = df['ingreso_bruto']

    # Crear el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Obtener coeficientes e intercepto
    coeficientes = model.coef_[0]
    intercepto = model.intercept_

    # Calcular predicciones
    y_pred = model.predict(X)

    # Calcular métricas de evaluación
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Crear un DataFrame para la línea de regresión
    df_regresion = pd.DataFrame({'Fecha Numerica': X['fecha_numerica'], 'Regresion': model.predict(X)})

    # Crear gráfico interactivo con Plotly
    fig = px.scatter(df, x='fecha_numerica', y='ingreso_bruto', title='Regresión Lineal')
    fig.add_scatter(x=df_regresion['Fecha Numerica'], y=df_regresion['Regresion'], mode='lines', name='Regresión lineal')

    # Convertir el gráfico a HTML
    graph_html = fig.to_html(full_html=False)

    return graph_html, {
        'coeficientes': coeficientes,
        'intercepto': intercepto,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
# Función de regresión neuronal
def realizar_regresion_neuronal(df):
    # Convertir fechas a valores numéricos (días desde el inicio)
    df['fecha_numerica'] = (pd.to_datetime(df['fecha']) - pd.to_datetime(df['fecha']).min()).dt.days

    X = df[['fecha_numerica']]
    y = df['ingreso_bruto']

    # Crear el modelo de regresión neuronal de dos capas
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Ajustar el modelo
    model.fit(X, y, epochs=100, verbose=0)

    # Obtener coeficientes e intercepto
    coeficientes = model.get_weights()[0]
    intercepto = model.get_weights()[1]

    # Calcular predicciones
    y_pred = model.predict(X)

    # Calcular métricas de evaluación
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Crear un DataFrame para la línea de predicción
    df_prediccion = pd.DataFrame({'Fecha Numerica': X['fecha_numerica'], 'Prediccion': y_pred.flatten()})

    # Crear gráfico interactivo con Plotly
    fig = px.scatter(df, x='fecha_numerica', y='ingreso_bruto', title='Regresión Neuronal')
    fig.add_scatter(x=df_prediccion['Fecha Numerica'], y=df_prediccion['Prediccion'], mode='lines', name='Predicción neuronal')

    # Convertir el gráfico a HTML
    graphR_html = fig.to_html(full_html=False)

    return graphR_html, {
        'coeficientes': coeficientes,
        'intercepto': intercepto,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Verificar si el formulario contiene un archivo
        if 'file' not in request.files:
            flash('No se encontró ningún archivo en el formulario.', 'error')
            return redirect(request.url)
        
        file = request.files['file']

        # Verificar si no se selecciona ningún archivo
        if file.filename == '':
            flash('No se seleccionó ningún archivo.', 'error')
            return redirect(request.url)

        # Verificar si el archivo es CSV
        if file and file.filename.endswith('.csv'):
            try:
                # Leer el archivo CSV y realizar operaciones necesarias
                df = pd.read_csv(file)
                # Calcular Ingreso Neto
                df['ingreso_neto'] = df['cantidad'] * df['precio_prod']

                # Calcular Margen de Ganancia
                df['margen_ganancia'] = (df['ingreso_neto'] / df['ingreso_bruto']) * 100

                # Calcular Crecimiento de Ingresos (ejemplo)
                df['crecimiento_ingresos'] = df['ingreso_neto'].pct_change() * 100

                # Calcular Totales
                tingresoN = df['ingreso_neto'].sum()
                tingresoB = df['ingreso_bruto'].sum()
                tmargenG = (tingresoN / tingresoB) * 100
                tcreceI = df['crecimiento_ingresos'].sum()

                # Contar registros
                count_registros = df.shape[0]

                # Sumar ingreso_bruto
                suma_ingreso_bruto = round(df['ingreso_bruto'].sum(), 3)

                # Sumar productos vendidos (cantidad)
                suma_productos_vendidos = int(df['cantidad'].sum())

                # Productos más vendidos
                productos_mas_vendidos = df.groupby('n_articulo').agg({'cantidad': 'sum', 'ingreso_bruto': 'sum'}).nlargest(3, 'cantidad').reset_index()
                #Productos menos vendidos
                productos_menos_vendidos = df.groupby('n_articulo').agg({'cantidad': 'sum', 'ingreso_bruto': 'sum'}).nsmallest(3, 'cantidad').reset_index()
                # Vendedores con más ventas
                vendedores_mas_ventas = df.groupby('vendedor').agg({'cantidad': 'sum', 'ingreso_bruto': 'sum'}).nlargest(3, 'cantidad').reset_index()

                # Clientes más frecuentes en términos de ingresos 
                clientes_mas_frecuentes = df.groupby('n_cliente').agg({'ingreso_bruto': 'sum'}).nlargest(3, 'ingreso_bruto').reset_index()

                # Almacenar los tops en variables
                pmvd = productos_mas_vendidos.to_dict(orient='records')
                pmvds = productos_menos_vendidos.to_dict(orient='records')
                vmvd = vendedores_mas_ventas.to_dict(orient='records')
                cmfd = clientes_mas_frecuentes.to_dict(orient='records')   

                # Calcular porcentajes para los tops
                porcentaje_pmvd = calcular_porcentaje_top_tres(productos_mas_vendidos, tingresoB)
                porcentaje_pmvds = calcular_porcentaje_top_tres(productos_menos_vendidos, tingresoB)
                porcentaje_vmvd = calcular_porcentaje_top_tres(vendedores_mas_ventas, tingresoB)
                porcentaje_cmfd = calcular_porcentaje_top_tres(clientes_mas_frecuentes, tingresoB)
                
                # Calcular la regresión lineal y obtener el gráfico interactivo con Plotly
                graph_html, resultados_lineal = realizar_regresion_lineal(df)
                graphR_html, resultados_neuronal = realizar_regresion_neuronal(df)

                flash('Archivo CSV cargado exitosamente.', 'success')
                return render_template('view.html',preview_data=df.head(), 
                                                graph_html=graph_html,
                                                graphR_html=graphR_html,
                                                resultados_lineal=resultados_lineal,
                                                resultados_neuronal=resultados_neuronal,
                                                count_registros=count_registros,
                                                suma_ingreso_bruto=suma_ingreso_bruto,
                                                suma_productos_vendidos=suma_productos_vendidos,
                                                ingresoN=tingresoN,
                                                margenG=tmargenG,
                                                creceI=tcreceI,
                                                productos_mas_vendidos=productos_mas_vendidos,
                                                productos_menos_vendidos=productos_menos_vendidos,
                                                vendedores_mas_ventas=vendedores_mas_ventas,
                                                clientes_mas_frecuentes=clientes_mas_frecuentes,
                                                pmvd=pmvd,
                                                pmvds=pmvds,
                                                vmvd=vmvd,
                                                cmfd=cmfd,
                                                porcentaje_pmvd=porcentaje_pmvd,
                                                porcentaje_pmvds=porcentaje_pmvds,
                                                porcentaje_vmvd=porcentaje_vmvd,
                                                porcentaje_cmfd=porcentaje_cmfd
                                            )

            except pd.errors.EmptyDataError:
                flash('El archivo CSV está vacío.', 'error')
                return render_template('form.html')
            except pd.errors.ParserError:
                flash('Error al analizar el archivo CSV. Asegúrate de que sea un archivo CSV válido.', 'error')
                return render_template('form.html')

        else:
            flash('Por favor, selecciona un archivo CSV.', 'error')
            return render_template('form.html')

    return render_template('form.html', preview_data=None)


if __name__ == '__main__':
    app.run(debug=True)
