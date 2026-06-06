import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

model = keras.models.load_model("dog_breed_model.keras")

df = pd.read_csv("labels.csv")

le = LabelEncoder()
le.fit(df["breed"])

correct = 0
total = 0

for breed_folder in os.listdir("test_image"):

    breed_path = os.path.join("test_image", breed_folder)

    if not os.path.isdir(breed_path):
        continue

    actual_breed = breed_folder.split("-", 1)[1].lower()

    for img_name in os.listdir(breed_path):

        img_path = os.path.join(breed_path, img_name)

        img = image.load_img(
            img_path,
            target_size=(128, 128)
        )

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction[0]
        )

        predicted_breed = le.inverse_transform(
            [predicted_class]
        )[0]

        if predicted_breed == actual_breed:
            correct += 1

        total += 1

print("\nEvaluation Results")
print("-" * 25)
print(f"Correct Predictions : {correct}")
print(f"Total Images        : {total}")
print(f"Accuracy            : {(correct / total) * 100:.2f}%")