import numpy as np
import timeit
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# Es wird das DenseNet201 Modell mit Standartgewichtung verwendet
model = DenseNet201(weights='imagenet')

# Anzahl Bilder
num_pic = 10

# num_pic Anzahl an Bildern wird eingelesen und vorbereitet
data = np.empty((num_pic, 224, 224, 3))

file = open('log_time_DN.txt','a')

for i in range(num_pic):
    data[i] = load_img('prepared_img224x224/' + str(i) + '.jpg')
data = preprocess_input(data) # das auch in funktion


def predict():
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