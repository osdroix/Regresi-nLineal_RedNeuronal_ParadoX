import pandas as pd

# Crear un DataFrame con tus datos
data = {
    'fecha': ['14/09/2020', '18/09/2020', '18/09/2020', '07/09/2020', '18/09/2020', '07/09/2020', '14/09/2020', '07/09/2020', '18/09/2020'],
    'cantidad': [2, 1, 4, 6, 2, 34, 2, 5, 20],
    'precio_prod': [974.99, 974.99, 974.99, 974.99, 974.99, 974.99, 974.99, 974.99, 974.99],
    'ingreso_bruto': [1949.98, 974.99, 3899.96, 5849.94, 1949.98, 33149.66, 1949.98, 4874.95, 19499.8]
}

df = pd.DataFrame(data)

# Calcular Ingreso Neto
df['ingreso_neto'] = df['cantidad'] * df['precio_prod']

# Calcular Margen de Ganancia
df['margen_ganancia'] = (df['ingreso_neto'] / df['ingreso_bruto']) * 100

# Calcular Crecimiento de Ingresos (ejemplo)
df['crecimiento_ingresos'] = df['ingreso_neto'].pct_change() * 100

# Calcular Totales
total_cantidad = df['cantidad'].sum()
total_ingreso_neto = df['ingreso_neto'].sum()
total_ingreso_bruto = df['ingreso_bruto'].sum()
total_margen_ganancia = (total_ingreso_neto / total_ingreso_bruto) * 100
total_crecimiento_ingresos = df['crecimiento_ingresos'].sum()

# Imprimir los Totales
print(f'Total Cantidad: {total_cantidad}')
print(f'Total Ingreso Neto: {total_ingreso_neto}')
print(f'Total Ingreso Bruto: {total_ingreso_bruto}')
print(f'Total Margen de Ganancia: {total_margen_ganancia}')
print(f'Total Crecimiento de Ingresos: {total_crecimiento_ingresos}')
