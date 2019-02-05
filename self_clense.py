import numpy as np
from keras.preprocessing import image
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
model = load_model("nsfw.299x299.h5")

# Empty lists to hold the images and labels for each each image
x_test = []
base_dir = 'D:\\nswf_model_training_data\\data'
group = 'test'
category = 'drawings'
file_type = "jpg"

# Load the data set by looping over every image file
for image_file in Path(base_dir + "\\" + group + "\\" +
                       category).glob("**/*." + file_type):

    # Load the current image file
    image_data = image.load_img(image_file, target_size=(299, 299))

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)

    # Add the current image to our list of test images
    x_test.append(image_array)

    # Add an expected label for this image. If it was a not_bird image, label it 0. If it was a bird, label it 1.
    # if "not_bird" in image_file.stem:
    #     y_test.append(0)
    # else:
    #     y_test.append(1)

# Convert the list of test images to a numpy array
x_test = np.array(x_test)

# Make predictions
predictions = model.predict(x_test)

print(predictions)

# predictions = predictions > 0.5

# # Calculate how many mis-classifications the model makes
# tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
# print(f"True Positives: {tp}")
# print(f"True Negatives: {tn}")
# print(f"False Positives: {fp}")
# print(f"False Negatives: {fn}")

# # Calculate Precision and Recall for each class
# report = classification_report(y_test, predictions)
# print(report)
