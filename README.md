# Beisbol_Predicción
## Predicción de la cantidad de juegos que puede ganar en una temporada un equipo de béisbol

También podemos ver el notebook aquí: https://nbviewer.jupyter.org/github/DanielAndres1116/Victorias_Beisbol_Prediccion/blob/main/PrediccionVictoriasBeisbol.ipynb 

### Descripción del Dataset y cómo se obtuvo
El dataset empleado en este proyecto se obtuvo de la página www.seanlahman.com. Sean Lahman es un reportero de investigación que ha recolectado un gran número de información sobre los equipos que forman parte de las ligas mayores de Estados Unidos. En el siguiente link podemos ver la información contenida en el dataset en cuestión: www.seanlahman.com/files/database/readme2017.txt

![image](https://user-images.githubusercontent.com/43154438/119208605-aacf2780-ba68-11eb-8a85-e08ac369fff0.png)

### Objetivos
Crear un modelo que determina cuántos juegos puede ganar en una temporada cualquier equipo de béisbol en las ligas mayores. Además de poner en práctica habilidades en procesamiento de datos y entrenamiento de modelos de machine learning para problemas de regresión, cabe destacar que aplicándole un modelo predictivo a estos datos es posible hacer una estimación de las características del juego que más influyen a la hora de obtener más victorias en la temporada para de esta manera optimizar los enfoques de entrenamiento de algún equipo en particular y entender los detalles más importantes en los que se debe mejorar. Entre estas características están: las carreras permitidas, carreras hechas, homeruns, bateos, hits, bases robadas, ponchadas, etc. Cabe resaltar que el principio de este análisis puede servir para enfocarlo a cualquier clase de deporte. 

#### Herramientas y librerías utilizadas
Se utiliza la librería de Pandas para importar el archivo correspondiente y gestionar los datos de manera adecuada antes de pasarlos por el modelo de Machine Learning. Es útil para hacer el análisis de cada una de las variables y eliminar columnas y/o filas con datos nulos, así como para hacer el preprocesamiento correspondiente de los datos y hacerlos más viables pare el modelo predictivo.
Se utiliza la librería de Matplotlib para hacer gráficos que permitan ir entendiendo los datos y visualizarlos por medio de histogramas, gráficos de dispersión y gráficos de unión de puntos. 
Se emplea la librería de sklearn para hacer la separación de datos de entrenamiento y prueba mediante el método train_test_split, para utilizar varios modelos de regresión, y para finalmente hacer una medida del error de los valores de la predicción con los reales mediante r2score y mean_squared_error y determinar el más útil.

Algunos de los gráficos de importancia realizados para entender los datos y ver su relevancia para el modelo fueron:


![image](https://user-images.githubusercontent.com/43154438/119208012-06e47c80-ba66-11eb-8ca7-16fdfbd1c8c3.png)

Figura 1: Este es un gráfico de dispersión que permite analizar qué variable tiene más peso sobre los juegos ganados, si las carreras permitidas o las hechas.

![image](https://user-images.githubusercontent.com/43154438/119208099-49a65480-ba66-11eb-9e81-9e9bfe126cb9.png)

Figura 2: En este histograma vemos el número de equipos que tuvo determinada cantidad de juegos ganados

![image](https://user-images.githubusercontent.com/43154438/119208190-99851b80-ba66-11eb-94ff-6b00a59eadb4.png)

Figura 3: Este gráfico lineal es para hecerse a una idea de cuánto varían las carreras realizadas a lo largo de los años

### Conclusiones y resultados obtenidos
Fueron utilizados varios algoritmos para verificar con cuál había mayor exactitud en la predicción. Entre estos algoritmos están: regresión lineal múltiple, regresión polinómica, máquinas de soporte vectorial para regresión, árbol de regresión y bosques aleatorios de regresión. Fueron utilizados los métodos de mean_squared_error y el de R cuadrado para hacer las respectivas observaciones y comparaciones del error en cada modelo. 

![image](https://user-images.githubusercontent.com/43154438/118034332-7a341300-b32f-11eb-81a9-e116aa2ef604.png)

Podemos ver que el modelo que mejor funcionó fue el de Regresión Lineal Múltiple con una exactitud de casi el 90%, y le sigue el modelo de Bosques Aleatorios de Regresión con casi un 70%. El resto de los modelos tuvieron un rendimiento muy regular.


