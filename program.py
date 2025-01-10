from ocr_segmentation import parse_image
from predict_single import predict_word
import os
import shutil
KERNEL_SENTENCE = (16,16)
KERNEL_WORD = (2,2)

image_path = r"TextImages\hello.png"
tmp_sentence_folder = r"tmp\sentence\words"
tmp_word_folder = r"tmp\word"
os.makedirs(tmp_sentence_folder, exist_ok=True)
parse_image(image_path, tmp_sentence_folder, KERNEL_SENTENCE, False) #Split image of sentence with no preprocess
sentence = ""
for file in os.listdir(tmp_sentence_folder):
    file = os.path.join(tmp_sentence_folder, file)
    # checking if it is a file

    if os.path.isfile(file):
        if os.path.exists(tmp_word_folder):
            shutil.rmtree(tmp_word_folder)
        os.makedirs(tmp_word_folder, exist_ok=True)
        parse_image(file, tmp_word_folder, KERNEL_WORD, True) #Split image of sentence with preprocess

        word = predict_word(tmp_word_folder)
        sentence = sentence + word + " "
        shutil.rmtree(tmp_word_folder)
print("Sentence recognized: " + sentence)
shutil.rmtree(tmp_sentence_folder)
