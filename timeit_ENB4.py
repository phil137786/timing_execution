import numpy as np
import timeit
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# Es wird das MobileNetV2 Modell mit Standartgewichtung verwendet
model = EfficientNetB4(weights='imagenet')

# Anzahl Bilder
num_pic = 10

# num_pic Anzahl an Bildern wird eingelesen und vorbereitet
data = np.empty((num_pic, 380, 380, 3))

file = open('log_time_ENB4.txt','a')

for i in range(num_pic):
    data[i] = load_img('prepared_img380x380/' + str(i) + '.jpg')
data = preprocess_input(data)


def predict():
    # Das Ergebnis wird ermittelt
    predictions = model.predict(data)

    # Ergebnisausgabe
    for i in range(num_pic): 
        output_neuron = np.argmax(predictions[i])

if __name__ == '__main__':
    Ausgabe=timeit.repeat("predict()", setup="from __main__ import predict", repeat=5, number =1)
    print (Ausgabe)
    average = sum(Ausgabe)/len(Ausgabe)
    print(file.write(str(Ausgabe)+" " + str(average) + "\n"))
    file.close()