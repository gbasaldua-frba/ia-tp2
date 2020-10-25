
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image

#---------------------------------------------------------------------------------#
# Definición de Parámetros
#---------------------------------------------------------------------------------#

#   Archivos del Modelo y Pesos
model_path = './model/model.json'
weights_model_path = './model/weights.h5'

#   Carpeta con datos de Entrenamiento y Test
train_data = './data/train' 
test_data = './data/test'


#---------------------------------------------------------------------------------#
# Obtención y Compilación del Modelo
#---------------------------------------------------------------------------------#

print("Carga del Modelo")

# se obtiene el modelos en formato json
json_file = open(model_path, 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

#   Se obtienen los pesos
model.load_weights(weights_model_path)

#   Compilacion del modelo con
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

#---------------------------------------------------------------------------------#
# Rutina para la evaluación del Modelo
#---------------------------------------------------------------------------------#

def probarModelo(x, y, clases_map):

    #   Se realiza la evaluación de cada imagen con el modelo generado 
    predClass = model.predict(x)

    #   Valor del umbral de aceptación
    umbralClas = 0.5
    classPreds = []
    classReal = []
    for i in range(len(x)):

        # prepara salida
        clReal = clases_map[ y[i] ]
        idclPred = predClass[i][0]

        ## determina clase predecida de acuerdo al umbral de clasificación
        idclPredRnd = int(idclPred)
        if (idclPred - idclPredRnd)>umbralClas and (idclPredRnd+1)<len(clases_map):
                idclPredRnd = idclPredRnd + 1

        if idclPredRnd<0 or idclPredRnd>=len(clases_map):
            clPred = "CLASE " + str(idclPredRnd) + " INVÁLIDA!"
        else:      
            clPred = clases_map[ idclPredRnd ]

        classReal.append( clReal )
        classPreds.append( clPred )

        # sólo muestra las imágenes no generadas por DA
        strTitulo = 'Real: ' + clReal + ' / RNA: ' 
        strTitulo = strTitulo + clPred + ' (' + str( idclPred ) +')'    

    # muestra reporte de clasificación
    print("\n Reporte de Clasificación: ")
    print(classification_report(classReal, classPreds))

    # muestra matriz de confusion
    print('\nMatriz de Confusión: ')
    cm = confusion_matrix(classReal, classPreds, labels=clases_map)
    cmtx = pd.DataFrame(
        cm, 
        index=['r:{:}'.format(x) for x in clases_map], 
        columns=['p:{:}'.format(x) for x in clases_map]
      )
    print(cmtx)
    print("\n")

    print("\n>Resultados: ")


#---------------------------------------------------------------------------------#
# Rutina para la carga de las Imágenes
#---------------------------------------------------------------------------------#

def cargarImagenes(imagPath):
  classes = []
  images = []

  all_dirs = os.listdir( imagPath )
  for each_dir in all_dirs:

      auxiPath = imagPath + '/' + each_dir 
      imagFN  = os.listdir( auxiPath )
      for each_imagFN in imagFN:

        #   Se abre la imagen a tratar
        imag = Image.open(auxiPath + "/" + each_imagFN)

        #   Se realiza el resizing de la imagen de acuerdo a la resolución
        # definida en los parámetros
        imag = imag.convert('RGB')
        imag = imag.resize((30, 30), Image.ANTIALIAS)

        #   Se genera el vector de la imagen para ser tratada por la capa 1
        arImag = np.array(imag)

        #   Se agrega el vector de la imagen junto con su clase
        classes.append( each_dir )
        images.append( arImag )

  return classes, images


#---------------------------------------------------------------------------------#
# Rutinas auxiliares para la carga de las Imágenes
#---------------------------------------------------------------------------------#

def prepare_imageList(imagList):
  auxiAr = np.array(imagList).astype('float32') / 255.
  auxiAr = auxiAr.reshape((len(auxiAr), 2700)) 
  return np.array(auxiAr)

def prepare_clasesList(classesList, dictMapeo=None):
  if dictMapeo==None:
    # genera diccionario de mapeo
    auxDict = list(set(classesList))
    dictMapeo = dict( zip( auxDict, range(len(auxDict)) ) )
  # realiza el mapeo
  y = []
  for cl in classesList:
      y.append( dictMapeo[cl] )
  return np.array(y), dictMapeo


#---------------------------------------------------------------------------------#
# Carga de Imágenes para la evaluación del Modelo
#---------------------------------------------------------------------------------#

#   Se cargan las imágenes del entrenamiento
classes_train, images_train = cargarImagenes(train_data)

#   Se cargan las imagenes para el testeo del Modelo
classes_test, images_test = cargarImagenes(test_data)

#   Se definen los vectores con el total de imágenes para el entrenamiento y test
x_train = prepare_imageList(images_train)
x_test = prepare_imageList(images_test)

# define vector auxiliar de datos de salida para usar en el entrenamiento y prueba
# también usa esta información para determinar la cantida de neuronas de salida
y_train, dictMapeo = prepare_clasesList(classes_train)
y_test, _ = prepare_clasesList(classes_test, dictMapeo)

# genera diccionario auxiliar para poder convertir de ID de clase a nombre de clase
clases_map = [ x for x,y in dictMapeo.items() ]

# prueba con los datos de entrenamiento
print("*** Resultados con datos de Entrenamiento: ")
probarModelo(x_train, y_train, clases_map)

# evalua al modelo entrenado
resEval = model.evaluate(x_test, y_test)
print("\n>Evaluación del Modelo: ")
print("    - Error: ", resEval[0])
print("    - Exactitud: ", resEval[1]*100)
print("\n")

# prueba con los datos de entrenamiento
print("\n\n*** Resultados con datos de Prueba: ")
probarModelo(x_test, y_test, clases_map)