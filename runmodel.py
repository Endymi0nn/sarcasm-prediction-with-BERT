
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
import os


absolute_path = os.path.dirname(__file__)
relative_path = "Bert_sarcasm.h5"
full_path = os.path.join(absolute_path, relative_path)


custom_objects = {'KerasLayer': hub.KerasLayer}

model=keras.models.load_model(full_path,custom_objects=custom_objects)

def pred(data):
    data=[data]
    if model.predict(data)>0.5:
        print("Sarcasm detected!")
    else:
        print("No sarcasm!")

while True:        
 sent=str(input("Your sentence here : "))
 if sent=='x':
     break
 else:
     pred(sent)
     print("\nMoving to next prediction .... To exit, input 'x'\n")
 


