import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report
from PIL import Image

#---------------------------------------------------------------------------------#
# Definición de Parámetros
#---------------------------------------------------------------------------------#

#   Archivos del Modelo y Pesos
model_path = './model/model.json'
weights_model_path = './model/weights.h5'

#---------------------------------------------------------------------------------#
# Obtención y Compilación del Modelo
#---------------------------------------------------------------------------------#

# se obtiene el modelos en formato json
json_file = open(model_path, 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

#   Se obtienen los pesos
model.load_weights(weights_model_path)

#   Compilacion del modelo con
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#   Se debe ingresar el path de la imagen
print("Ingrese el path de la Image: ")
img_path = input()

#   Se obtiene la imagen de la PC
img_color = Image.open(img_path)
img=img_color

#   Se da el formato necesario para ser evaluado por el modelo
img = img.convert('RGB')
img = img.resize((30, 30), Image.ANTIALIAS)
img_vec = np.array(img)
auxiAr = img_vec.astype('float32') / 255.
auxiAr = auxiAr.reshape(1, 2700)


predict = model.predict(auxiAr)

if predict > 0.5:
    result='Hay un Perro en la Imagen!'
else:
    result='Sin Mascotas!'

plt.title(result)
plt.imshow(img_color)
plt.show()