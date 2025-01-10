import cv2
import os
import numpy as np
#import deeplake



def sort_contours(contours):
    # Extract bounding boxes and sort by y (row-wise)
    bounding_boxes = [(cv2.boundingRect(c), c) for c in contours]
    bounding_boxes.sort(key=lambda b: b[0][1])  # Sort by y-coordinate (vertical position)

    # Group contours by rows based on y-threshold (row separation)
    rows = []
    row_threshold = 20  # You can adjust this as needed
    current_row = []
    prev_y = bounding_boxes[0][0][1]  # Get the y of the first bounding box

    for bbox, contour in bounding_boxes:
        x, y, w, h = bbox
        if abs(prev_y - y) < row_threshold:  # If within same row
            current_row.append((x, y, w, h, contour))
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))  # Sort by x (horizontal position)
            current_row = [(x, y, w, h, contour)]  # New row
        prev_y = y

    if current_row:  # Don't forget to add the last row
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # Flatten the sorted rows into a single list of contours
    return [contour for row in rows for _, _, _, _, contour in row]


def parse_image(image_path, output_folder, kernel_size, preprocess):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours ,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sort_contours(contours)
    i = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        word_image = image[y:y + h, x:x + w]

        padding = 5  # Amount of padding you want around the word
        word_image = cv2.copyMakeBorder(
            word_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)  # Black padding
        )

        output_path = os.path.join(output_folder, f'word_{i}.png')
        if preprocess: word_image = cv2.bitwise_not(word_image)  # if preprocess active

        cv2.imwrite(output_path, word_image)
        i += 1


    print("Words saved to:", output_folder)



