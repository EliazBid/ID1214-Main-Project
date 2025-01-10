import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

#pip install emnist
from emnist import list_datasets
list_datasets()


#I extract training samples from the EMNIST dataset.
#The 'balanced' argument indicates the extraction of a balanced subset of data.
#It includes an equal number of samples for each character class.

from emnist import extract_training_samples
x_train, y_train = extract_training_samples('balanced')
from emnist import extract_test_samples
x_test, y_test = extract_test_samples('balanced')
x_train.shape, y_train.shape, x_test.shape, y_test.shape

print("Number of training examples:", len(x_train))
print("Number of test examples:", len(x_test))
print("Image dimensions:", x_train[0].shape)
print("Number of classes:", len(np.unique(y_train)))

x_train=x_train/255
x_test=x_test/255

#array NumPy
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_test =  np.array(x_test).reshape(-1, 28, 28, 1)
print(x_train.shape)
print(x_test.shape)


y_train = to_categorical(y_train)


K.clear_session()

# Define the model using the functional approach
cnn= Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='tanh', input_shape=(28, 28, 1)))
cnn.add(MaxPooling2D(strides=2))
cnn.add(Conv2D(filters=48, kernel_size=(5,5), padding='same', activation='tanh'))
cnn.add(MaxPooling2D( strides=2))
cnn.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='tanh'))
cnn.add(Flatten())
cnn.add(Dense(512, activation='tanh'))
cnn.add(Dense(84, activation='tanh'))
cnn.add(Dense(47, activation='softmax'))

#optimazer
opt=Adam(learning_rate=1e-4)

#Let's compile the model before training.
cnn.compile(optimizer=opt,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

cnn.summary()

keras.utils.plot_model(cnn, "model.png", show_shapes=True)


#callback

keras_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.001)
]

start_time = time.time()
y_train_cat = np.argmax(y_train, axis=-1)  # Convert from (112800, 47, 2) to class labels (112800, 47)
y_train_cat = to_categorical(y_train_cat, num_classes=47)
print(y_train_cat.shape)
history = cnn.fit(x_train, y_train_cat, epochs=100, batch_size=64, verbose=1,
                    validation_split=0.2, callbacks=keras_callbacks)
cnn.save("EMNISTModel.h5")  # Saves the model in HDF5 format
end_time = time.time()

val_losses=history.history["val_loss"]
train_losses=history.history["loss"]

epochs = range(1, len(val_losses) + 1)

plt.figure()
plt.title("Training loss")
plt.plot(epochs,val_losses,c="red",label="Validation")
plt.plot(epochs,train_losses,c="orange",label="Training")
plt.xlabel("Epochs")
plt.ylabel("Cross entropy")
plt.legend()