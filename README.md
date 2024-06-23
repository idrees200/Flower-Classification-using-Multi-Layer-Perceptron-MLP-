# Flower Classification using Multi-Layer Perceptron (MLP)

This repository contains code for a Multi-Layer Perceptron (MLP) model trained to classify flower images into five categories: tulip, sunflower, rose, dandelion, and daisy. The MLP is implemented from scratch using NumPy and OpenCV for image processing. The dataset is provided in a ZIP file containing images of different flower species.

## Dataset
The dataset consists of flower images stored in a ZIP file named `flowers.zip`. Each image is resized to 100x100 pixels and normalized to have pixel values between 0 and 1. Images are loaded, processed, and split into training, validation, and test sets using the `train_test_split` function from scikit-learn.

## Model Architecture
The MLP model has the following architecture:
- Input Layer: 30,000 neurons (100x100x3 for RGB channels)
- Hidden Layers: Two hidden layers with 128 and 64 neurons respectively, activated using the sigmoid function.
- Output Layer: Five neurons (corresponding to five flower categories), activated using the softmax function.

## Training
The model is trained using backpropagation with a cross-entropy loss function and stochastic gradient descent. Training occurs over multiple epochs, and during each epoch, the training and validation losses are computed to monitor model performance. The learning rate is set to 0.01, and batch size is 32.

## Evaluation
After training, the model's performance is evaluated on the test set. Metrics such as accuracy, classification report, and Receiver Operating Characteristic (ROC) curve are used to assess the model's effectiveness in flower classification.

## Files Included
- `MLP.py`: Contains the MLP class definition for training and prediction.
- `flowers.zip`: Dataset containing flower images.
- `README.md`: This file, providing an overview of the project, dataset, model architecture, training process, and evaluation metrics.

## Instructions to Run
1. Download or clone the repository.
2. Ensure Python 3.x and required libraries (NumPy, OpenCV, scikit-learn, Matplotlib) are installed.
3. Extract `flowers.zip` into the project directory.
4. Run `MLP.py` to train the model and evaluate its performance.

## Results
The model achieves an accuracy of approximately [insert accuracy score] on the test set. See the classification report in `MLP.py` for detailed metrics.

## Further Improvements
- Experiment with different network architectures (e.g., more layers, different activation functions).
- Explore data augmentation techniques to improve model generalization.
- Implement hyperparameter tuning to optimize model performance.


![Screenshot 2024-06-23 191145](https://github.com/idrees200/Flower-Classification-using-Multi-Layer-Perceptron-MLP-/assets/113856749/ba8cefaa-b0f1-46c3-a9b8-247b51e06321)
![Screenshot 2024-06-23 191139](https://github.com/idrees200/Flower-Classification-using-Multi-Layer-Perceptron-MLP-/assets/113856749/cc60d526-7433-4269-adb4-59332f87a551)
![Screenshot 2024-06-23 191134](https://github.com/idrees200/Flower-Classification-using-Multi-Layer-Perceptron-MLP-/assets/113856749/b92a9241-c803-4732-a8f3-d0eee5cc58f1)
![Screenshot 2024-06-23 190214](https://github.com/idrees200/Flower-Classification-using-Multi-Layer-Perceptron-MLP-/assets/113856749/1d2c6d09-1742-4fbf-944b-d538c6a3f0da)
![Screenshot 2024-06-23 191151](https://github.com/idrees200/Flower-Classification-using-Multi-Layer-Perceptron-MLP-/assets/113856749/69807435-57bf-4663-ad05-b5e8dfcb04e6)
