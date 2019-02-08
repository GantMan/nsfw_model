import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing import image
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

model = load_model("nsfw.299x299.h5")
test_dir = 'D:\\nswf_model_training_data\\data\\test'
image_size = 299
x_test = []
y_test = []
file_count = 0
update_frequency = 1000

class_names = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

for image_file in Path(test_dir).glob("**/*.jpg"):
    file_count += 1
    # Load the current image file
    image_data = image.load_img(image_file, target_size=(image_size, image_size))

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)
    image_array /= 255

    # Add to list of test images
    x_test.append(image_array)
    # Now add answer derived from folder
    path_name = os.path.dirname(image_file)
    folder_name = os.path.basename(path_name)
    y_test.append(class_names.index(folder_name))

    if file_count % update_frequency == 0:
        print("Processed " + str(file_count) + " - Current Folder: " + folder_name)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

x_test = np.array(x_test)
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()