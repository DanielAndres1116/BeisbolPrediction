# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:31:13 2020

@author: Daniel Andres
"""

### IMPORTAR LIBRERÍAS ###
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### IMPORTAR LOS DATOS ###
data = pd.read_csv('Teams.csv')

### ANALIZAR LOS DATOS ###
#Conocer la forma de los datos 
tamano=data.shape

#Conocer el formato de los datos
tiposdatos=data.dtypes

#Conocer los datos nulos
datosnulos=data.isnull().sum()

drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin',
             'WCWin','LgWin','WSWin','SF','name','park','attendance',
             'BPF','PPF','teamIDBR','teamIDlahman45','teamIDretro']

data = data.drop(drop_cols, axis=1)

#Elimino columnas con datos nulos 
datosnulos=data.isnull().sum()
data = data.drop(['CS','HBP'], axis=1)

#Calculo la media para completar los datos nulos
data['BB'] = data['BB'].fillna(data['BB'].median())
data['SO'] = data['SO'].fillna(data['SO'].median())
data['SB'] = data['SB'].fillna(data['SB'].median())

datosnulos=data.isnull().sum()

#Separo los años por eras en las etapas del baseball
i = 0
for year in data['yearID']:
    if year < 1920:
        data.loc[i, "era"] = 1
    elif year >= 1920 and year <= 1941:
        data.loc[i,"era"] = 2
    elif year >= 1942 and year <= 1945:
        data.loc[i,"era"] = 3
    elif year >= 1946 and year <= 1962:
        data.loc[i,"era"] = 4    
    elif year >= 1963 and year <= 1976:
        data.loc[i,"era"] = 5
    elif year >= 1977 and year <= 1992:
        data.loc[i,"era"] = 6
    elif year >= 1993 and year <= 2009:
        data.loc[i,"era"] = 7
    elif year >= 2010:
        data.loc[i,"era"] = 8
    i +=1
    
#Separo los años por eras en las etapas por décadas
j = 0
for year in data['yearID']:
    if year < 1920:
        data.loc[j, "decada"] = 1910
    elif year >= 1920 and year <= 1929:
        data.loc[j,"decada"] = 1920
    elif year >= 1930 and year <= 1939:
        data.loc[j,"decada"] = 1930
    elif year >= 1940 and year <= 1949:
        data.loc[j,"decada"] = 1940  
    elif year >= 1950 and year <= 1959:
        data.loc[j,"decada"] = 1950
    elif year >= 1960 and year <= 1969:
        data.loc[j,"decada"] = 1960
    elif year >= 1970 and year <= 1979:
        data.loc[j,"decada"] = 1970
    elif year >= 1980 and year <= 1989:
        data.loc[j,"decada"] = 1980
    elif year >= 1990 and year <= 1999:
        data.loc[j,"decada"] = 1990
    elif year >= 2000 and year <= 2009:
        data.loc[j,"decada"] = 2000
    elif year >= 2010:
        data.loc[j,"decada"] = 2010
    j +=1
    
### SELECCIONO EL EQUIPO DE BASEBALL A EVALUAR ###
team = 'NYA'

data_team = data.loc[data.teamID == team]

### VISUALIZACIÓN DE LOS DATOS ###
#Visualizar los juegos ganados
plt.hist(data_team['W'])
plt.xlabel('Juegos ganados')
plt.title('Distribución de los Juegos Ganados')
plt.show()

#Visualizar las carreras realizadas por cada año
plt.plot(data_team['yearID'], data_team['R'])
plt.title('Carreras Realizadas por Año')
plt.xlabel('Año')
plt.ylabel('Carreras Realizadas')
plt.show()

#Visualizar el número de carreras realizadas y permitidas por juegos ganados
#Determino el estimado de carreras realizadas por cada juego 
R_x_juego = data_team['R'] / data_team['G']
RA_x_juego = data_team['RA'] / data_team['G']

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.scatter(R_x_juego, data_team['W'], c='blue')
ax1.set_title('Carreras por juegos vs. Juegos Ganados')
ax1.set_ylabel('Juegos Ganados')
ax1.set_xlabel('Carreras por juegos')
ax2.scatter(RA_x_juego, data_team['W'], c='red')
ax2.set_title('Carreras Permitidas por Juegos vs Juegos Ganados')
ax2.set_xlabel('Carreras Permitidas por Juegos')
plt.show()

#Elimino las columnas que no son necesarias
data_team = data_team.drop(['yearID','teamID'], axis = 1)


### ANÁLISIS DE MACHINE LEARNING ###
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


y = data_team['W']
X = data_team.drop('W', axis = 1)

#Separar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)

#Definir el algoritmo 
algoritmoLR = LinearRegression()
algoritmoPol2 = LinearRegression()
algoritmoPol3 = LinearRegression()
poli_reg2 = PolynomialFeatures(degree=2)
poli_reg3 = PolynomialFeatures(degree=3)
svr1 = SVR(kernel='linear', C=1.0, epsilon=0.1)
svr2 = SVR(kernel='rbf', degree=3, C=1.0, epsilon=0.1)
adr = DecisionTreeRegressor(max_depth=500)
bar = RandomForestRegressor(n_estimators=300, max_depth=10)


#Ajustar el orden de las variables de entrada para la regresión polinómica
X_train_poli2 = poli_reg2.fit_transform(X_train)
X_test_poli2 = poli_reg2.fit_transform(X_test)
X_train_poli3 = poli_reg3.fit_transform(X_train)
X_test_poli3 = poli_reg3.fit_transform(X_test)


#Entrenar el algoritmo
algoritmoLR.fit(X_train, y_train)
algoritmoPol2.fit(X_train_poli2, y_train)
algoritmoPol3.fit(X_train_poli3, y_train)
svr1.fit(X_train, y_train)
svr2.fit(X_train, y_train)
adr.fit(X_train, y_train)
bar.fit(X_train, y_train)

#Realizar una predicción
y_test_predLR = algoritmoLR.predict(X_test)
y_test_predPol2 = algoritmoPol2.predict(X_test_poli2)
y_test_predPol3 = algoritmoPol3.predict(X_test_poli3)
y_test_predsvr1 = svr1.predict(X_test)
y_test_predsvr2 = svr2.predict(X_test)
y_test_predadr = adr.predict(X_test)
y_test_predBar = bar.predict(X_test)

#Calculo de la precisión del modelo
#Error promedio al cuadrado
rmse_rf = (mean_squared_error(y_test,y_test_predLR))**(1/2)
print("Error Promedio al cuadrado con Regresión Lineal: ", rmse_rf)
rmse_rf_bar = (mean_squared_error(y_test,y_test_predBar))**(1/2)
print("Error Promedio al cuadrado con Forest Decision Regression: ", rmse_rf_bar)
rmse_rf_Pol3 = (mean_squared_error(y_test,y_test_predPol3))**(1/2)
print("Error Promedio al cuadrado con Regresión Polinómica de orden 3: ", rmse_rf_Pol3)
rmse_rf_Pol2 = (mean_squared_error(y_test,y_test_predPol2))**(1/2)
print("Error Promedio al cuadrado con Regresión Polinómica de orden 2: ", rmse_rf_Pol2)
rmse_rf_adr = (mean_squared_error(y_test,y_test_predadr))**(1/2)
print("Error Promedio al cuadrado con Tree Decision Regression: ", rmse_rf_adr)
rmse_rf_svr1 = (mean_squared_error(y_test,y_test_predsvr1))**(1/2)
print("Error Promedio al cuadrado con Support Vector Regression: ", rmse_rf_svr1)


#Calculo R2
print("R2 Regresión Lineal: ", r2_score(y_test, y_test_predLR))
print("R2 Forest Decision Regression: ", r2_score(y_test, y_test_predBar))
print("R2 Regresión Polinómica Orden 3: ", r2_score(y_test, y_test_predPol3))
print("R2 Regresión Polinomica Orden 2: ", r2_score(y_test, y_test_predPol2))
print("R2 Tree Decision Regression: ", r2_score(y_test, y_test_predadr))
print("R2 Support Vector Regression Kernel Lineal: ", r2_score(y_test, y_test_predsvr1))


