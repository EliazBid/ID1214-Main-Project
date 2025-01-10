from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from emnist import extract_test_samples
from PIL import Image
import os

class_labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't', 47: '!', 48: '?', 49: '%',
    50: '@', 51: '#', 52: '&', 53: '$'
}

x_test_emnist, y_test_emnist = extract_test_samples('balanced')
x_test_emnist = x_test_emnist / 255.0 
x_test_emnist = x_test_emnist.reshape(-1, 28, 28, 1)  
cnn = load_model('CombinedModel.h5')

# variables for the custom symbols
custom_symbols_folder = "generated_symbols_test"
custom_symbols = ['!', '?', '%', '@', '#', '&', '$']
num_images_per_symbol = 400

# arrays for custom symbol storage
custom_images = []
custom_labels = []

emnist_classes = 47 

for counter, file_name in enumerate(sorted(os.listdir(custom_symbols_folder), key=lambda x: int(x.split('.')[0]))):
    
    # open custom image
    image_path = os.path.join(custom_symbols_folder, file_name)
    image = Image.open(image_path).convert('L')
    image_array = np.array(image) / 255.0 

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
    custom_images.append(image_array.reshape(28, 28, 1))
    custom_labels.append(label)

custom_images = np.array(custom_images)
custom_labels = np.array(custom_labels)

# Combine EMNIST and custom test data
x_test_combined = np.concatenate([x_test_emnist, custom_images], axis=0)
y_test_combined = np.concatenate([y_test_emnist, custom_labels], axis=0)

prob = cnn.predict(x_test_combined)
y_pred = np.argmax(prob, axis=-1)

cm = confusion_matrix(y_test_combined, y_pred)

plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 


tick_marks = np.arange(len(class_labels)) 
plt.xticks(tick_marks + 0.5, [class_labels[i] for i in range(len(class_labels))]) 
plt.yticks(tick_marks + 0.5, [class_labels[i] for i in range(len(class_labels))])

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Combined')
plt.show()