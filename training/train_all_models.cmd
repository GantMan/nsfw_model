:: You can add more models types from here: https://tfhub.dev/s?module-type=image-classification&tf-version=tf2
:: However, you must choose Tensorflow 2 models. V1 models will not work here.
:: https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4
:: https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4
:: https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4
:: https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4
::
:: If you get CUDA_OUT_OF_MEMORY crash, you need to pass --batch_size NUMBER, reducing until you don't get this error.
:: It is advised by Google not to have a batch size < 8.

:: Note that we set all of our target epochs to over 9000. This is because the trainer just uses early stopping internally.

:: Train Mobilenet V2 140
python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\mobilenet_v2_140_224 --labels_output_file %cd%\..\trained_models\mobilenet_v2_140_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4 --tflite_output_file %cd%\..\trained_models\mobilenet_v2_140_224\saved_model.tflite --train_epochs 9001 --batch_size 32 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\mobilenet_v2_140_224 --labels_output_file %cd%\..\trained_models\mobilenet_v2_140_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4 --tflite_output_file %cd%\..\trained_models\mobilenet_v2_140_224\saved_model.tflite --train_epochs 9001 --batch_size 32 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
:: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\mobilenet_v2_140_224 %cd%\..\trained_models\mobilenet_v2_140_224\web_model
:: Or, for a quantized (1 byte) version
:: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\mobilenet_v2_140_224 %cd%\..\trained_models\mobilenet_v2_140_224\web_model_quantized --quantization_bytes 1

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\efficientnet_b0 --labels_output_file %cd%\..\trained_models\efficientnet_b0\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b0\saved_model.tflite --train_epochs 9001 --batch_size 16 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\efficientnet_b0 --labels_output_file %cd%\..\trained_models\efficientnet_b0\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b0\saved_model.tflite --train_epochs 9001 --batch_size 16 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 240 --saved_model_dir %cd%\..\trained_models\efficientnet_b1 --labels_output_file %cd%\..\trained_models\efficientnet_b1\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b1\saved_model.tflite --train_epochs 9001 --batch_size 12 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 240 --saved_model_dir %cd%\..\trained_models\efficientnet_b1 --labels_output_file %cd%\..\trained_models\efficientnet_b1\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b1\saved_model.tflite --train_epochs 9001 --batch_size 12 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 260 --saved_model_dir %cd%\..\trained_models\efficientnet_b2 --labels_output_file %cd%\..\trained_models\efficientnet_b2\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b2\saved_model.tflite --train_epochs 9001 --batch_size 10 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 260 --saved_model_dir %cd%\..\trained_models\efficientnet_b2 --labels_output_file %cd%\..\trained_models\efficientnet_b2\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b2\saved_model.tflite --train_epochs 9001 --batch_size 10 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 300 --saved_model_dir %cd%\..\trained_models\efficientnet_b3 --labels_output_file %cd%\..\trained_models\efficientnet_b3\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b3\saved_model.tflite --train_epochs 9001 --batch_size 8 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 300 --saved_model_dir %cd%\..\trained_models\efficientnet_b3 --labels_output_file %cd%\..\trained_models\efficientnet_b3\class_labels.txt --tfhub_module https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1 --tflite_output_file %cd%\..\trained_models\efficientnet_b3\saved_model.tflite --train_epochs 9001 --batch_size 8 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

:: Train Resnet V2 50
python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\resnet_v2_50_224 --labels_output_file %cd%\..\trained_models\resnet_v2_50_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4 --tflite_output_file %cd%\..\trained_models\resnet_v2_50_224\saved_model.tflite --train_epochs 9001 --batch_size 16 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\resnet_v2_50_224 --labels_output_file %cd%\..\trained_models\resnet_v2_50_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4 --tflite_output_file %cd%\..\trained_models\resnet_v2_50_224\saved_model.tflite --train_epochs 9001 --batch_size 16 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
::tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\resnet_v2_50_224 %cd%\..\trained_models\resnet_v2_50_224\web_model
:: Or, for a quantized (1 byte) version
::tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\resnet_v2_50_224 %cd%\..\trained_models\resnet_v2_50_224\web_model_quantized --quantization_bytes 1

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

:: Train Inception V3
python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\inception_v3_224 --labels_output_file %cd%\..\trained_models\inception_v3_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4 --tflite_output_file %cd%\..\trained_models\inception_v3_224\saved_model.tflite --train_epochs 9001 --batch_size 16 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\inception_v3_224 --labels_output_file %cd%\..\trained_models\inception_v3_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4 --tflite_output_file %cd%\..\trained_models\inception_v3_224\saved_model.tflite --train_epochs 9001 --batch_size 16 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
::tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\inception_v3_224 %cd%\..\trained_models\inception_v3_224\web_model
:: Or, for a quantized (1 byte) version
::tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\inception_v3_224 %cd%\..\trained_models\inception_v3_224\web_model_quantized --quantization_bytes 1

:: Wait for Python/CUDA/GPU to recover. Seems to die without this.
Timeout /T 60 /Nobreak

:: Train NasNetMobile
python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\nasnet_a_224 --labels_output_file %cd%\..\trained_models\nasnet_a_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4 --tflite_output_file %cd%\..\trained_models\nasnet_a_224\saved_model.tflite --train_epochs 9001 --batch_size 24 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0

python make_nsfw_model.py --image_dir %cd%\..\images --image_size 224 --saved_model_dir %cd%\..\trained_models\nasnet_a_224 --labels_output_file %cd%\..\trained_models\nasnet_a_224\class_labels.txt --tfhub_module https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4 --tflite_output_file %cd%\..\trained_models\nasnet_a_224\saved_model.tflite --train_epochs 9001 --batch_size 24 --do_fine_tuning --dropout_rate 0.0 --label_smoothing=0.0 --validation_split=0.1 --do_data_augmentation=True --use_mixed_precision=True --rmsprop_momentum=0.0 --learning_rate=0.0005

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
::tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\nasnet_a_224 %cd%\..\trained_models\nasnet_a_224\web_modely
:: Or, for a quantized (1 byte) version
::tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\nasnet_a_224 %cd%\..\trained_models\nasnet_a_224\web_modely_quantized --quantization_bytes 1