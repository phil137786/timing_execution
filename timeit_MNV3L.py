import numpy as np
import timeit
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# Es wird das MobileNetV2 Modell mit Standartgewichtung verwendet
model = MobileNetV3Large(weights='imagenet')

# Anzahl Bilder
num_pic = 10

# num_pic Anzahl an Bildern wird eingelesen und vorbereitet
data = np.empty((num_pic, 224, 224, 3))

file = open('log_time_MNV3L.txt','a')

for i in range(num_pic):
    data[i] = load_img('prepared_img224x224/' + str(i) + '.jpg')

def predict():
    # Vorbereiten muss auch gemessen werden
    data = preprocess_input(data) 

    # Das Ergebnis wird ermittelt
    predictions = model.predict(data)

    # Ergebnisausgabe
    for i in range(num_pic): 
        output_neuron = np.argmax(predictions[i])

if __name__ == '__main__':
    import timeit
    Ausgabe=timeit.repeat("predict()", setup="from __main__ import predict", repeat=5, number =10)
    print (Ausgabe)
    average = sum(Ausgabe)/len(Ausgabe)
    print(file.write(str(Ausgabe)+" " + str(average) + "\n"))
    file.close()