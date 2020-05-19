:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
::
:: Note that at present, the efficientnet models will not successful convert because
:: we're waiting on an update in tfjs to reach release for that support.
::
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\mobilenet_v2_140_224 %cd%\..\trained_models\mobilenet_v2_140_224\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\mobilenet_v2_140_224 %cd%\..\trained_models\mobilenet_v2_140_224\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b0 %cd%\..\trained_models\efficientnet_b0\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b0 %cd%\..\trained_models\efficientnet_b0\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b1 %cd%\..\trained_models\efficientnet_b1\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b1 %cd%\..\trained_models\efficientnet_b1\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b2 %cd%\..\trained_models\efficientnet_b2\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b2 %cd%\..\trained_models\efficientnet_b2\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b3 %cd%\..\trained_models\efficientnet_b3\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\efficientnet_b3 %cd%\..\trained_models\efficientnet_b3\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\resnet_v2_50_224 %cd%\..\trained_models\resnet_v2_50_224\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\resnet_v2_50_224 %cd%\..\trained_models\resnet_v2_50_224\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\inception_v3_224 %cd%\..\trained_models\inception_v3_224\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\inception_v3_224 %cd%\..\trained_models\inception_v3_224\web_model_quantized --quantization_bytes 1

:: Note that installing tensorflowjs also installs tensorflow-cpu A.K.A. bye-bye-training. So make sure you perform this step after all your training is done, and then restore a GPU version of TF.
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\nasnet_a_224 %cd%\..\trained_models\nasnet_a_224\web_model
:: Or, for a quantized (1 byte) version
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve %cd%\..\trained_models\nasnet_a_224 %cd%\..\trained_models\nasnet_a_224\web_model_quantized --quantization_bytes 1