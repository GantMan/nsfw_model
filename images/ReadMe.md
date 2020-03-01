# Images Setup

Place a folder with images inside of it for each class you want. The training scripts will automatically separate train/test/validation from each
folder and will generate a new label for each folder. The labels will be written to the output model directory. Consult the .cmd and .sh training
scripts in the training folder for full usage in the event that you are invoking the python scripts directly.

You should pre-resize your images to the target network input size. Otherwise, your training times will be increased by several hours due to the
preprocessing expense. Anecdotally, there is little to no benefit to runtime transforms during this kind of transfer learning and fine-tuning learning.