# Wild-Animal-Image-Classifier
  
## Project Overview
Wild-Animal-Image-Classifier is a deep learning project for classifying images of wild animals using advanced neural network architectures such as VGG19, ResNet, Bi-LSTM, and Bi-GRU. The project includes data augmentation techniques to improve model robustness against noisy and far images.

## Features
- Image classification of wild animals
- Data augmentation for improved generalization
- Multiple model architectures: VGG19, ResNet, VGG19+Bi-LSTM, VGG19+Bi-GRU
- Evaluation metrics: accuracy, precision, recall, F1-score

## Files
- `wild_animal_vgg19,resnet,vgg19_+bi_lstm_,vgg19+bi_gru.py`: Main script for training and evaluating models
- `WildAnimalDetection.ipynb`: Jupyter notebook for experimentation and analysis

## Usage
1. Prepare your dataset and update the paths in the scripts as needed.
2. Run the Python script or Jupyter notebook to train and evaluate models:
	```sh
	python wild_animal_vgg19,resnet,vgg19_+bi_lstm_,vgg19+bi_gru.py
	```
	or open `WildAnimalDetection.ipynb` in Jupyter and run the cells.

## Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- scikit-learn

Install dependencies with:
```sh
pip install tensorflow keras numpy opencv-python scikit-learn
```

## License
This project is for educational purposes.