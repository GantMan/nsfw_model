from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from time import time
from tensorflow_model_optimization.sparsity import keras as sparsity

# Slow down training deeper into dataset
def schedule(epoch):
    if epoch < 6:
        # Warmup model first
        return .0000032
    elif epoch < 12:
        return .01
    elif epoch < 20:
        return .002
    elif epoch < 40:
        return .0004
    elif epoch < 60:
        return .00008
    elif epoch < 80:
        return .000016
    elif epoch < 95:
        return .0000032        
    else:
        return .0000009       


def make_callbacks(weights_file):
    current_log_dir = "logs/{}".format(time())
    # checkpoint
    filepath = weights_file
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Update info
    #tensorboard = TensorBoard(log_dir=current_log_dir)

    # learning rate schedule
    lr_scheduler = LearningRateScheduler(schedule)

    # all the goodies
    return [
        lr_scheduler, 
        checkpoint, 
        #tensorboard,
        sparsity.UpdatePruningStep(),
        #sparsity.PruningSummaries(log_dir=current_log_dir, profile_batch=0)
    ]