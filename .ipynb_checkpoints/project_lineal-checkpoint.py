# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:47:31 2023

@author: osdroix
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sns.set_style('darkgrid')

# Leer datos
datos = pd.read_csv('DATAVH.csv')

# Tomar una muestra del 10% de los datos
datos_sample = datos.sample(frac=0.1, random_state=1)

# Seleccionar columnas relevantes
nuevo = datos_sample[['fecha', 'cantidad', 'actividad_economica']]

# Generar pairplot
g = sns.pairplot(nuevo, hue='fecha', diag_kind='hist')
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=45)

# Mostrar el plot
plt.show()
