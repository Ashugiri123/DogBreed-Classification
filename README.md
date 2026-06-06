# Dog Breed Classification

## Overview

This project is a Dog Breed Classification System built using TensorFlow and InceptionV3 Transfer Learning.

The model predicts the breed of a dog from an input image.

The system was trained on 120 dog breeds and evaluated on an external dataset containing unseen dog images.

---

## Features

* Classifies 120 dog breeds
* Uses Transfer Learning with InceptionV3
* Supports single-image prediction
* Provides Top-5 predictions
* Includes automated evaluation script

---

## Project Structure

DogBreed/

├── dogBreed.ipynb

├── predict.py

├── evaluate.py

├── dog_breed_model.keras

├── labels.csv

├── requirements.txt

├── train/

├── test/

└── test_image/

---

## Model

* Architecture: InceptionV3 Transfer Learning
* Input Size: 128 × 128 × 3
* Output Classes: 120 Dog Breeds

---

## Evaluation

External Evaluation Dataset: 20,580 Images

Accuracy: 64.38%

The model was evaluated on an external dataset containing unseen dog images from 120 dog breeds.

---

## Installation

pip install -r requirements.txt

---

## Run Prediction

python predict.py image.jpg

---

## Example Output

Top 5 Predictions:

papillon : 82.13%
chihuahua : 10.24%
toy_terrier : 4.83%
pekinese : 1.72%
pomeranian : 0.56%

---

## Run Evaluation

python evaluate.py

---

## Technologies Used

* Python
* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* VS Code
