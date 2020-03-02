import os
import numpy as np
from keras.preprocessing import image
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Initialize
model = load_model("nsfw.299x299.h5")
image_size = 299
file_count = 0
x_test = []
y_test = []
mistakes = []
categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
base_dir = 'D:\\nswf_model_training_data\\data'
batch_size = 1000
page = 0

# CONFIGURE EACH RUN
group = 'train'
category_id = 4
mistaken_as = 2
file_type = "jpg"


def process_batch(batch_x, batch_y):
    print("Batch Check " + str(file_count))
    # Convert the list of images to a numpy array
    x_array = np.array(batch_x)

    # Make predictions (arrays of size 5, with probabilities)
    predictions = model.predict(x_array)
    max_predictions = np.argmax(predictions, axis=1)

    for idx, prediction in enumerate(max_predictions):
        if prediction != category_id:
            # We have a mistake!  Do we log it?
            if prediction == mistaken_as:
                mistakes.append(batch_y[idx])

# Copies categorization failures to the mistakes folder for analysis
def copy_all_failures():
    for file_info in mistakes:
        os.rename(file_info["path"], base_dir + "\\" + group + "\\mistakes\\" + str(file_info["filename"]))

print("Starting Self-clense for " + categories[category_id])
# Load the data set by looping over every image file in path
for image_file in Path(base_dir + "\\" + group + "\\" +
                       categories[category_id]).glob("**/*." + file_type):
    file_info = {"path": image_file, "filename": os.path.basename(image_file)}

    top = (page + 1) * batch_size
    file_count += 1

    # Load the current image file
    image_data = image.load_img(image_file, target_size=(image_size, image_size))

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)
    image_array /= 255

    # Add the current image to our list of test images
    x_test.append(image_array)
    # To identify failed predictions
    y_test.append(file_info)

    # Kick off a processing to clear RAM
    if file_count == top:
        process_batch(x_test, y_test)
        # move next batch moment
        page += 1
        # reset in-memory
        x_test = []
        y_test = []

process_batch(x_test, y_test)     
copy_all_failures()   
print('Out of ' + str(file_count) + ' images of "' + str(categories[category_id]) + '" ' + str(len(mistakes)) + ' are mistaken as "' + str(categories[mistaken_as]) + '"')