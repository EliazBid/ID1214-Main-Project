from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import random

# custom symbols
custom_symbols = ['!', '?', '%', '@', '#', '&', '$']

# number of images per symbol, same as for the EMNIST balanced dataset
num_images_per_symbol = 2400

# folder for the custom symbols
output_folder = "generated_symbols"
os.makedirs(output_folder, exist_ok=True)

def create_symbol_image(symbol):

    image_size=(28, 28) 
    font_size = random.randint(15, 25)  # random font size for symbolising different writings of the same symbol
    background = Image.new('L', image_size, color=0)  # 28x28 black background
    image = ImageDraw.Draw(background) # image object
    font = ImageFont.truetype("arial.ttf", font_size)

    # calculate the width and height of the symbol
    bbox = image.textbbox((0, 0), symbol, font=font) # generates a tuple of (left, top, right, bottom)
    text_width = bbox[2] - bbox[0] # right - left
    text_height = bbox[3] - bbox[1] # bottom - top

    # find the center to make sure character is in image
    position = ((image_size[0] - text_width) // 2, 
                (image_size[1] - text_height) // 2)

    # add the symbol
    image.text(position, symbol, fill=255, font=font)

    # to symbolise different writings of the same symbol, use rotation
    rotation_angle = random.randint(-30, 30)
    rotated_image = background.rotate(rotation_angle, expand=True)

    # extract the center region of the rotated image
    rotated_width, rotated_height = rotated_image.size
    left = (rotated_width - image_size[0]) // 2
    top = (rotated_height - image_size[1]) // 2
    right = left + image_size[0]
    bottom = top + image_size[1]

    # crop to the original image size
    cropped_image = rotated_image.crop((left, top, right, bottom))

    return np.array(cropped_image)

# generate the symbols to the generated_symbols folder, roughly 5 MB large, can take a while finish executing
image_count = 0
for i, symbol in enumerate(custom_symbols):
    for j in range(num_images_per_symbol):
        symbol_image = create_symbol_image(symbol)
        image_path = os.path.join(output_folder, f"{image_count + 1}.png") # named 1.png, 2.png and so on
        Image.fromarray(symbol_image).save(image_path)
        image_count += 1

print(f"Generated {num_images_per_symbol * len(custom_symbols)} symbol images are saved in the folder: {output_folder}")
