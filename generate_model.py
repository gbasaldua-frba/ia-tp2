
import tensorflow as tf
import tensorflow.keras
import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
from PIL import Image


#---------------------------------------------------------------------------------#
# Inicialización de la Sesión
#---------------------------------------------------------------------------------#
K.clear_session()


#---------------------------------------------------------------------------------#
# Definición de Parámetros
#---------------------------------------------------------------------------------#

#   Se define la resolución de las imágenes para el entrenamiento, en este
# caso se realizará un resizing a una resolución de 30x30 en formato RGB
IMAGE_SHAPE = (30, 30, 3)

#   Se define el número de neuronas de la capa de Input como el producto de la
# resolución por los tres colores del formato RGB
num_inputs = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]

#   Se define la salida, en este caso dos correspondientes a cada clase
num_outputs = 1

#   Se definen un total de cuatro capas ocultas con la cantidad de neuronas en
# cada capa de 540, 135, 27 y 9
hidden_layers = [ 540, 135, 27 , 9 ]

#   Cantidad de épocas del entrenamiento
cantEpocas = 300

#   Valor del batch size para el entrenamiento
batchSize = 15

#   Carpetas de Entrenamiento, Prueba y Guardado del modelo
train_data = './data/train' 
test_data = './data/test'
target_dir = './model/'


#---------------------------------------------------------------------------------#
# Carga de Imágenes de Entrenamiento y Test
#---------------------------------------------------------------------------------#

#   Rutina para la carga de imágenes de un directorio
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
        imag = imag.resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), Image.ANTIALIAS)

        #   Se genera el vector de la imagen para ser tratada por la capa 1
        arImag = np.array(imag)

        #   Se agrega el vector de la imagen junto con su clase
        classes.append( each_dir )
        images.append( arImag )

  return classes, images

#   Se cargan las imágenes para el entrenamiento
classes_train, images_train = cargarImagenes(train_data)

# Rutina auxiliar para el armado de la lista de imágenes a procesar
def prepare_imageList(imagList):
  auxiAr = np.array(imagList).astype('float32') / 255.
  auxiAr = auxiAr.reshape((len(auxiAr), num_inputs))  
  return np.array(auxiAr)

#   Rutina auxiliar para el armado de la lista de clases 
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

#   Se definen los vectores con el total de imágenes para el entrenamiento
x_train = prepare_imageList(images_train)

#   Se definen las clases para el Entrenamiento
y_train, dictMapeo = prepare_clasesList(classes_train)

#   Se definen las leyendas de las Clases del Modelo
clases_map = [ x for x,y in dictMapeo.items() ]


#---------------------------------------------------------------------------------#
# Armado del Modelo
#---------------------------------------------------------------------------------#

#    Se define la capa de Entrada
input_img_Lay = Input(shape=(num_inputs,), name='input_img')
eachLay = input_img_Lay
auxName = 'hidd_'
auxId = 1 
for num_hid in hidden_layers:  
    
    #   Se definen y agregan cada una de las capas ocultas
    auxlayerName = auxName+str(auxId)
    auxId = auxId + 1
    eachLay = Dense(num_hid, name=auxlayerName)(eachLay) # capas ocultas

#   Se define la capa de Salida
output_img_Lay = Dense(num_outputs, activation=None, name='output')(eachLay) # capa de salida

#   Se genera el modelo RNA MLP Backpropagation
model = Model(input_img_Lay, output_img_Lay, name='RNA')
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#   Se muestra el resumen del Modelo generado
model.summary()
print("\n")


#---------------------------------------------------------------------------------#
# Entrenamiento del Modelo
#---------------------------------------------------------------------------------#

#   Se realiza el entrenamiento del Modelo con los vectores de datos generados y
# los parámetros de cantidad de épocas y bach size definidos
history = model.fit(x_train, y_train,
                    epochs = cantEpocas,
                    batch_size = batchSize)

print("Fin de entrenamiento del modelo")


#---------------------------------------------------------------------------------#
# Guardado del Modelo
#---------------------------------------------------------------------------------#

#   Se crea la carpeta donde se almacenará el Modelo y sus Pesos
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

#   Se genera el Modelo en formato JSON
model_jason = model.to_json()

#   Se almacena el Modelo generado en formato JSON y los Pesos en formato H5
with open("./model/model.json", "w") as json_file:
    json_file.write(model_jason)

#   Se almacenan los pesos en formato H5
model.save_weights('./model/weights.h5')

print("Se ha guardado el modelo generado")


#---------------------------------------------------------------------------------#
# Se generan los gráficos del Entrenamiento
#---------------------------------------------------------------------------------#

#   Se genera el gráfico de avance de la precisión según avanza el Entrenamiento
# del modelo y se almacena en la carpeta del mismo
plt.figure()
plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.xlabel('# epochs')
plt.ylabel('accurancy')
plt.legend()
plt.savefig('./model/accur.png')
plt.close()

#   Se genera el gráfico de avance de la pérdida según avanza el Entrenamiento
# del modelo y se almacena en la carpeta del mismo
plt.figure()
plt.plot(history.history['loss'],'r',label='training loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./model/loss.png')
plt.close()