# Vibrational-Signal-Classification
This project is centered around two key objectives: dimensionality reduction and vibration data classification. We employ both autoencoder and variational autoencoder techniques to reduce dimensionality. Subsequently, the extracted low-dimensional representations are employed for classification, with available options including SVM, Neural Network, and Logistic Regression.

## Table of Contents
1. Autoencoder Architecture
2. Training the Autoencoder
3. Reconstruction and Evaluation



# Autoencoder Architecture

The autoencoder architecture consists of an encoder and decoder. The encoder reduces the dimensionality of the input signal, and the decoder reconstructs the signal from the encoded representation.
```python
# Code for SignalEncoder class
from tensorflow import keras
from tensorflow.keras import layers

class SignalEncoder(keras.Model):
    def __init__(self, input_dim):
        super(SignalEncoder, self).__init__()
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(400, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu")
        ])

        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(50,)),
            layers.Dense(100, activation="relu"),
            layers.Dense(400, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

```

# Training the Autoencoder
The autoencoder is trained using the normalized training data.
```python
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(train_data_normalized, train_data_normalized,
                          epochs=20,
                          batch_size=512,
                          validation_data=(validation_data_normalized, validation_data_normalized),
                          shuffle=True)

```
 # Reconstruction and Evaluation
 The trained autoencoder is used to reconstruct the test data, and the reconstruction error is evaluated
```python
reconstructions = autoencoder.predict(test_data_normalized)
test_loss = tf.keras.losses.mae(reconstructions, test_data_normalized)

```
# SVM Classification
The SVM models are created using the SVC class from scikit-learn. The Polynomial kernel is implemented with a degree of 8, and the RBF kernel is used with the "auto" setting for gamma.
```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Function to get the corresponding SVC model based on the kernel type
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernel
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function (RBF) kernel
        return SVC(kernel='rbf', gamma="auto")

# Training and evaluating SVM models
for i in range(2):
    svclassifier = getClassifier(i)
    svclassifier.fit(train_encoded_data, y_train)
    y_pred = svclassifier.predict(test_encoded_data)
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test, y_pred))

```
# Grid Search for Hyperparameter Tuning
A grid search is performed using the GridSearchCV class from scikit-learn. This helps find the optimal hyperparameters for the SVM model.
The grid search explores various combinations of the regularization parameter (C), gamma, and kernel type.
```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly']}

# Create a grid search object
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

# Fit the grid search to the data
grid.fit(train_encoded_data, y_train)

# Print the best parameters after tuning
print("Best parameters:", grid.best_params_)

# Print the model with the best hyperparameters
print("Best model:", grid.best_estimator_)

```
# Evaluation

The performance of each SVM model is evaluated using classification reports.
Classification reports provide insights into precision, recall, and F1-score for each class

![Capture d'écran 2023-11-30 164101](https://github.com/Mayssabr/Vibrational-Signal-Classification/assets/80195974/494c4b7c-281e-41cf-a31a-7afcf49f1182)

