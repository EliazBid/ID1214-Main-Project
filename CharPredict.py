from tensorflow.keras.preprocessing import image
import numpy as np
from keras.models import load_model
import os

class_labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't', 47: '!', 48: '?', 49: '%',
    50: '@', 51: '#', 52: '&', 53: '$'
}
cnn = load_model(r'CombinedModel.h5')

def predict_single_input(img_path):
    # Load the image and preprocess
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0

    # Expand dimensions to match the batch size (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    # Predict using the trained model
    prob = cnn.predict(img_array)
    y_pred = np.argmax(prob, axis=-1)

    # Get the predicted label
    predicted_label = class_labels[y_pred[0]]
    prediction_probability = prob[0][y_pred[0]]  # This will give the probability for the predicted class


    return predicted_label, prediction_probability


def predict_word(word_dir):
    word = ""
    for filename in os.listdir(word_dir):
        f = os.path.join(word_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            predicted_label, prediction_probability = predict_single_input(f)
            word = word + predicted_label
            print(f"Predicted Label: {predicted_label}")
            print(f"Prediction Probability: {prediction_probability * 100:.2f}%")  # Display probability as a percentage
    return word