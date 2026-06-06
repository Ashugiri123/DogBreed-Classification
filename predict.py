from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import sys

MODEL_PATH = "dog_breed_model.keras"
model = keras.models.load_model(MODEL_PATH)
df = pd.read_csv("labels.csv")

le = LabelEncoder()
le.fit(df["breed"])

if len(sys.argv) != 2:
    print("Usage: python predict.py image.jpg")
    sys.exit()
img_path = sys.argv[1]

img = image.load_img(
    img_path,
    target_size=(128, 128)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array, verbose=0)

top5 = np.argsort(prediction[0])[-5:][::-1]

print("Top 5 Predictions:\n")

for idx in top5:
    breed = le.inverse_transform([idx])[0]
    conf = prediction[0][idx] * 100

    print(f"{breed} : {conf:.2f}%")