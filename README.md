# Vibrational-Signal-Classification
This project is centered around two key objectives: dimensionality reduction and vibration data classification. We employ both autoencoder and variational autoencoder techniques to reduce dimensionality. Subsequently, the extracted low-dimensional representations are employed for classification, with available options including SVM, Neural Network, and Logistic Regression.

## Table of Contents
1. [Autoencoder Architecture](#Autoencoder Architecture)
2. [Training the Autoencoder](#Training the Autoencoder)
3. [Reconstruction and Evaluation](#Reconstruction and Evaluation)



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



