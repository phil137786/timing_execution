import numpy as np
from tensorflow.keras.applications import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input
from keras.applications.mobilenet_v3 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img

# Es wird das MobileNetV3Large Modell mit Standartgewichtung verwendet
model = MobileNetV3Large(weights='imagenet')

data = np.empty((1, 224, 224, 3))
data[0] = load_img('prepared_img224x224/' + str(0) + '.jpg')
data = preprocess_input(data)

# Das Ergebnis wird ermittelt
predictions = model.predict(data)

# Ergebnisausgabe
output_neuron = np.argmax(predictions[0])
print('Most active neuron: {} ({:.2f}%)'.format(output_neuron, 100 * predictions[0][output_neuron]))
for name, desc, score in decode_predictions(predictions)[0]:
    print('- {} ({:.2f}%%)'.format(desc, 100 * score))











