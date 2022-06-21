import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img

# Es wird das MobileNetV2 Modell mit Standartgewichtung verwendet
model = MobileNetV2(weights='imagenet')

# Anzahl Bilder
num_pic = 10

# num_pic Anzahl an Bildern wird eingelesen und vorbereitet
data = np.empty((num_pic, 224, 224, 3))

for i in range(num_pic):
    data[i] = load_img('prepared_img224x224/' + str(i) + '.jpg')
data = preprocess_input(data) 


# Das Ergebnis wird ermittelt
predictions = model.predict(data)

# Ergebnisausgabe
for i in range(num_pic):
    output_neuron = np.argmax(predictions[i])
    print('Most active neuron: {} ({:.2f}%)'.format(output_neuron, 100 * predictions[i][output_neuron]))
    for name, desc, score in decode_predictions(predictions)[i]:
        print('- {} ({:.2f}%%)'.format(desc, 100 * score))