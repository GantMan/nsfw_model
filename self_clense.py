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
accepted_file_count = 0
mistake_count = 0
x_test = []
y_test = []
categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
base_dir = 'D:\\nswf_model_training_data\\data'
batch_size = 8000

# CONFIGURE EACH RUN
group = 'train'
category_id = 1
file_type = "jpg"
page = 1


bottom = page * batch_size
top = (page + 1) * batch_size
# Load the data set by looping over every image file
for image_file in Path(base_dir + "\\" + group + "\\" +
                       categories[category_id]).glob("**/*." + file_type):
    file_info = {"path": image_file, "filename": os.path.basename(image_file)}

    file_count += 1

    # limit results for in-memmory batches
    if bottom <= file_count < top:
        accepted_file_count += 1
        # Load the current image file
        image_data = image.load_img(image_file, target_size=(image_size, image_size))

        # Convert the loaded image file to a numpy array
        image_array = image.img_to_array(image_data)
        image_array /= 255

        # Add the current image to our list of test images
        x_test.append(image_array)
        # To identify failed predictions
        y_test.append(file_info)

# Convert the list of test images to a numpy array
x_test = np.array(x_test)

# Make predictions (arrays of size 5, with probabilities)
predictions = model.predict(x_test)
# print(predictions)
max_predictions = np.argmax(predictions, axis=1)

for idx, prediction in enumerate(max_predictions):
    if prediction != category_id:
        mistake_count += 1
        # print('Thought it was ' + categories[prediction] + ': ' + str(y_test[idx]["path"]))
        os.rename(y_test[idx]["path"], base_dir + "\\" + group + "\\mistakes\\" + str(y_test[idx]["filename"]))
        
print('Out of ' + str(accepted_file_count) + ' images of "' + str(categories[category_id]) + '" ' + str(mistake_count) + ' are failures')