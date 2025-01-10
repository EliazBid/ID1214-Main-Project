import os
from PIL import Image
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
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

x_train = x_train / 255.0
x_test = x_test / 255.0

#array NumPy
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_test =  np.array(x_test).reshape(-1, 28, 28, 1)
print(x_train.shape)
print(x_test.shape)

y_train = to_categorical(y_train)


# Number of EMNIST classes
emnist_classes = y_train.shape[1]

print("EMNIST Training Data:", x_train.shape, y_train.shape)

# variables for the custom symbols
custom_symbols_folder = "generated_symbols"
custom_symbols = ['!', '?', '%', '@', '#', '&', '$']
num_images_per_symbol = 2400

# arrays for custom symbol storage
custom_images = []
custom_labels = []

# read files in order 1, 2, 3,...2300, 2301, etc   
for counter, file_name in enumerate(sorted(os.listdir(custom_symbols_folder), key=lambda x: int(x.split('.')[0]))):
    
    # open custom image
    img_path = os.path.join(custom_symbols_folder, file_name)
    img = Image.open(img_path).convert('L')  
    img_array = np.array(img) / 255.0  
    
    # every num_images_per_symbol images, we start a new symbol and a new label
    if counter < num_images_per_symbol:
        label = emnist_classes  # start after EMNIST classes
    elif counter < num_images_per_symbol * 2:
        label = emnist_classes + 1
    elif counter < num_images_per_symbol * 3:
        label = emnist_classes + 2
    elif counter < num_images_per_symbol * 4:
        label = emnist_classes + 3
    elif counter < num_images_per_symbol * 5:
        label = emnist_classes + 4
    elif counter < num_images_per_symbol * 6:
        label = emnist_classes + 5
    elif counter < num_images_per_symbol * 7:
        label = emnist_classes + 6
    # add additional elif statements if more custom symbols are used
    
    # add the image and label the the arrays
    custom_images.append(img_array.reshape(28, 28, 1))
    custom_labels.append(label)

# turn custom symbol arrays into numpy arrays, same as EMNIST arrays
custom_images = np.array(custom_images)
custom_labels = to_categorical(custom_labels, num_classes=emnist_classes + len(custom_symbols))

print("Custom Symbols Data:", custom_images.shape, custom_labels.shape)

# increase the label num from EMNISTs 47 to account for how many custom symbols we have
y_train_padded = np.pad(y_train, ((0, 0), (0, len(custom_symbols))), mode='constant')

# combine the datasets
x_train_combined = np.concatenate([x_train, custom_images], axis=0)
y_train_combined = np.concatenate([y_train_padded, custom_labels], axis=0)

# shuffle the dataset
shuffle_indices = np.random.permutation(len(x_train_combined))
x_train_combined = x_train_combined[shuffle_indices]
y_train_combined = y_train_combined[shuffle_indices]

print("Combined Training Data Shape:", x_train_combined.shape, y_train_combined.shape)

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
cnn.add(Dense(emnist_classes + len(custom_symbols), activation='softmax'))  # output layer

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

history = cnn.fit(x_train_combined, y_train_combined, epochs=100, batch_size=64, verbose=1,
                  validation_split=0.2, callbacks=keras_callbacks)
cnn.save("CombinedModel.h5") # Saves the model in HDF5 format

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
plt.show()
