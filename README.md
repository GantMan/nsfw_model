![NSFW Detector logo](https://github.com/GantMan/nsfw_model/blob/master/_art/nsfw_detection.png?raw=true)

# NSFW Detection Machine Learning Model

[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)

Trained on 60+ Gigs of data to identify:
- `drawings` - safe for work drawings (including anime)
- `hentai` - hentai and pornographic drawings
- `neutral` - safe for work neutral images
- `porn` - pornographic images, sexual acts
- `sexy` - sexually explicit images, not pornography

This model powers [NSFW JS](https://github.com/infinitered/nsfwjs) - [More Info](https://shift.infinite.red/avoid-nightmares-nsfw-js-ab7b176978b1)

## Current Status:

93% Accuracy with the following confusion matrix, based on Inception V3.
![nsfw confusion matrix](_art/nsfw_confusion93.png)

## Requirements:

See [requirements.txt](requirements.txt).

## Usage

For programmatic use of the library. 

```python
from nsfw_detector import predict
model = predict.load_model('./nsfw_mobilenet2.224x224.h5')

# Predict single image
predict.classify(model, '2.jpg')
# {'2.jpg': {'sexy': 4.3454722e-05, 'neutral': 0.00026579265, 'porn': 0.0007733492, 'hentai': 0.14751932, 'drawings': 0.85139805}}

# Predict multiple images at once
predict.classify(model, ['/Users/bedapudi/Desktop/2.jpg', '/Users/bedapudi/Desktop/6.jpg'])
# {'2.jpg': {'sexy': 4.3454795e-05, 'neutral': 0.00026579312, 'porn': 0.0007733498, 'hentai': 0.14751942, 'drawings': 0.8513979}, '6.jpg': {'drawings': 0.004214506, 'hentai': 0.013342537, 'neutral': 0.01834045, 'porn': 0.4431829, 'sexy': 0.5209196}}

# Predict for all images in a directory
predict.classify(model, '/Users/bedapudi/Desktop/')

```

If you've installed the package or use the command-line this should work, too...

```sh
# a single image
nsfw-predict --saved_model_path mobilenet_v2_140_224 --image_source test.jpg

# an image directory
nsfw-predict --saved_model_path mobilenet_v2_140_224 --image_source images

# a single image (from code/CLI)
python3 nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg

```


## Download
Please feel free to use this model to help your products!  

If you'd like to [say thanks for creating this, I'll take a donation for hosting costs](https://www.paypal.me/GantLaborde).

# Latest Models Zip (v1.1.0)
https://github.com/GantMan/nsfw_model/releases/tag/1.1.0

### Original Inception v3 Model (v1.0)
* [Keras 299x299 Image Model](https://s3.amazonaws.com/ir_public/ai/nsfw_models/nsfw.299x299.h5)
* [TensorflowJS 299x299 Image Model](https://s3.amazonaws.com/ir_public/ai/nsfw_models/nsfwjs.zip)
* [TensorflowJS Quantized 299x299 Image Model](https://s3.amazonaws.com/ir_public/ai/nsfw_models/min_nsfwjs.zip)
* [Tensorflow 299x299 Image Model](https://s3.amazonaws.com/ir_public/ai/nsfw_models/nsfw.299x299.pb) - [Graph if Needed](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#inspecting-graphs)

### Original Mobilenet v2 Model (v1.0)
* [Keras 224x224 Image Model](https://s3.amazonaws.com/ir_public/nsfwjscdn/nsfw_mobilenet2.224x224.h5)
* [TensorflowJS 224x224 Image Model](https://s3.amazonaws.com/ir_public/nsfwjscdn/TFJS_nsfw_mobilenet/tfjs_nsfw_mobilenet.zip)
* [TensorflowJS Quantized 224x224 Image Model](https://s3.amazonaws.com/ir_public/nsfwjscdn/TFJS_nsfw_mobilenet/tfjs_quant_nsfw_mobilenet.zip)
* [Tensorflow 224x224 Image Model](https://s3.amazonaws.com/ir_public/nsfwjscdn/TF_nsfw_mobilenet/nsfw_mobilenet.pb) - [Graph if Needed](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#inspecting-graphs)
* [Tensorflow Quantized 224x224 Image Model](https://s3.amazonaws.com/ir_public/nsfwjscdn/TF_nsfw_mobilenet/quant_nsfw_mobilenet.pb) - [Graph if Needed](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#inspecting-graphs)

## PyTorch Version
Kudos to the community for creating a PyTorch version with resnet!
https://github.com/yangbisheng2009/nsfw-resnet

## TF1 Training Folder Contents
Simple description of the scripts used to create this model:
* `inceptionv3_transfer/` - Folder with all the code to train the Keras based Inception v3 transfer learning model.  Includes `constants.py` for configuration, and two scripts for actual training/refinement.
* `mobilenetv2_transfer/` - Folder with all the code to train the Keras based Mobilenet v2 transfer learning model.
* `visuals.py` - The code to create the confusion matrix graphic
* `self_clense.py` - If the training data has significant inaccuracy, `self_clense` helps cross validate errors in the training data in reasonable time.   The better the model gets, the better you can use it to clean the training data manually.

_e.g._
```bash
cd training
# Start with all locked transfer of Inception v3
python inceptionv3_transfer/train_initialization.py

# Continue training on model with fine-tuning
python inceptionv3_transfer/train_fine_tune.py

# Create a confusion matrix of the model
python visuals.py
```

## Extra Info
There's no easy way to distribute the training data, but if you'd like to help with this model or train other models, get in touch with me and we can work together.

Advancements in this model power the quantized TFJS module on https://nsfwjs.com/

My Twitter is [@GantLaborde](https://twitter.com/GantLaborde) - I'm a School Of AI Wizard New Orleans.  I run the twitter account [@FunMachineLearn](https://twitter.com/FunMachineLearn)

Learn more about [me](http://gantlaborde.com/) and the [company I work for](https://infinite.red/).

Special thanks to the [nsfw_data_scraper](https://github.com/alexkimxyz/nsfw_data_scrapper) for the training data.  If you're interested in a more detailed analysis of types of NSFW images, you could probably use this repo code with [this data](https://github.com/EBazarov/nsfw_data_source_urls).

If you need React Native, Elixir, AI, or Machine Learning work, check in with us at [Infinite Red](https://infinite.red/), who make all these experiments possible.  We're an amazing software consultancy worldwide!

## Cite
```
@misc{man,
  title={Deep NN for NSFW Detection},
  url={https://github.com/GantMan/nsfw_model},
  journal={GitHub},
  author={Laborde, Gant}}
```

## Contributors

Thanks goes to these wonderful people ([emoji key](https://github.com/kentcdodds/all-contributors#emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
| [<img src="https://avatars0.githubusercontent.com/u/997157?v=4" width="100px;"/><br /><sub><b>Gant Laborde</b></sub>](http://gantlaborde.com/)<br />[💻](https://github.com/GantMan/nsfw_model/commits?author=GantMan "Code") [📖](https://github.com/GantMan/nsfw_model/commits?author=GantMan "Documentation") [🤔](#ideas-GantMan "Ideas, Planning, & Feedback") | [<img src="https://avatars2.githubusercontent.com/u/15898654?v=4" width="100px;"/><br /><sub><b>Bedapudi Praneeth</b></sub>](http://bpraneeth.com)<br />[💻](https://github.com/GantMan/nsfw_model/commits?author=bedapudi6788 "Code") [🤔](#ideas-bedapudi6788 "Ideas, Planning, & Feedback") |
| :---: | :---: |
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification. Contributions of any kind welcome!
