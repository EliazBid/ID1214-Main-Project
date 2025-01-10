from tkinter import Label, Button, Toplevel,filedialog, messagebox, Entry
from tkinterdnd2 import TkinterDnD, DND_FILES
from ocr_segmentation import parse_image
from CharPredict import predict_word
import os
import shutil

KERNEL_SENTENCE = (16, 16)
KERNEL_WORD = (3,3)

def process_image(image_path):
    tmp_sentence_folder = r"tmp\sentence\words"
    tmp_word_folder = r"tmp\word"
    os.makedirs(tmp_sentence_folder, exist_ok=True)

    try:
        parse_image(image_path, tmp_sentence_folder, KERNEL_SENTENCE, False)  # Split image of sentence with no preprocess
        sentence = ""
        for file in os.listdir(tmp_sentence_folder):
            file = os.path.join(tmp_sentence_folder, file)
            if os.path.isfile(file):
                if os.path.exists(tmp_word_folder):
                    shutil.rmtree(tmp_word_folder)
                os.makedirs(tmp_word_folder, exist_ok=True)
                parse_image(file, tmp_word_folder, KERNEL_WORD, True)  # Split image of sentence with preprocess

                word = predict_word(tmp_word_folder)
                sentence += word + " "
                shutil.rmtree(tmp_word_folder)
        messagebox.showinfo("Recognized Sentence", f"Sentence: {sentence.strip()}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        shutil.rmtree(tmp_sentence_folder)

def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")],
        title="Select an Image File"
    )
    if file_path:
        process_image(file_path)

def handle_drop(event):
    file_path = event.data
    if os.path.isfile(file_path):
        process_image(file_path)
    else:
        messagebox.showerror("Error", "Invalid file. Please drop a valid image file.")
def open_settings():
    settings_window = Toplevel(root)
    settings_window.title("Kernel Settings")
    settings_window.geometry("500x300")
    Label(settings_window, text="Try smaller kernel sizes for sentences with small spaces").pack(pady=5)
    Label(settings_window, text="Sentence Kernel (Height, Width):").pack(pady=5)

    sentence_width_entry = Entry(settings_window)
    sentence_width_entry.insert(0, str(KERNEL_SENTENCE[0]))
    sentence_width_entry.pack(pady=5)

    sentence_height_entry = Entry(settings_window)
    sentence_height_entry.insert(0, str(KERNEL_SENTENCE[1]))
    sentence_height_entry.pack(pady=5)
    Label(settings_window, text="Try smaller kernel sizes for words with small spaces between characters").pack(pady=5)

    Label(settings_window, text="Word Kernel (Height, Width):").pack(pady=5)

    word_width_entry = Entry(settings_window)
    word_width_entry.insert(0, str(KERNEL_WORD[0]))
    word_width_entry.pack(pady=5)

    word_height_entry = Entry(settings_window)
    word_height_entry.insert(0, str(KERNEL_WORD[1]))
    word_height_entry.pack(pady=5)

    def save_settings():
        try:
            global KERNEL_SENTENCE, KERNEL_WORD
            KERNEL_SENTENCE = (int(sentence_width_entry.get()), int(sentence_height_entry.get()))
            KERNEL_WORD = (int(word_width_entry.get()), int(word_height_entry.get()))
            messagebox.showinfo("Settings Saved", "Kernel sizes updated")
            settings_window.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integers for kernel sizes.")

    Button(settings_window, text="Save", command=save_settings).pack(pady=20)

# Create GUI
root = TkinterDnD.Tk()
root.title("Handwritten character detection")
root.geometry("640x160")

label = Label(root, text="Drag and drop an image file below or click 'Select Image'.")
label.pack(pady=10)

button = Button(root, text="Select Image", command=select_file)
button.pack(pady=20)

button_settings = Button(root, text="Settings", command=open_settings)
button_settings.pack(pady=5)


#Enables drag and drop support
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', handle_drop)

root.mainloop()
