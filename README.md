# Kaggle YouTube-8M Video Understanding Challenge

This repo contains a code for the 2nd YouTube8M Kaggle Challenge. It is extension of starter code provided by Google:[link](https://github.com/google/youtube-8m).

For the challenge we used 3 different neural network architectures:

**1.  Simple 2 layer neural network:**

The input to the network is 1152 dim vector. First dense layer compresses it to 576 dim i.e. half of the input dim, so that it can learn the correlation between different dimensions. The output is then passed to final dense layer of size 3862 i.e. number of classes of video, followed by sigmoid activation function.

**2.  Branched 'V'-shaped neural network:**

As the input contains visual features as well as audio features, we decided to separate and process them individually. In this model, we split the input 1152 dim vector into 1024 dim visual vector and 128 dim vector. Give the two split vectors to two separate dense layer to reduce the dimensions to half, creating vectors of visual feature of size 512 and audio feature of size 64. Then we concatenate these vectors and give it to dense layer and then to softmax layer for video classification.

**3.  Convolutional Neural Network:**

In this model, instead of concatenating the two feature vectors, we decided to calculate outer product, which gives us a matrix and we applied 2 layer Convolutional Neural Network for capturing the pattern. The features are then flatten and given to softmax layer for classification.


*video_level_models.py* contains the implementation of above mentioned models. For all the models we used Cross-entropy as loss function.


## Download fraction of video level data:

There are two types data given for the challenge. The one is video-level data, which contains visual and audio features, that are averaged over 300 seconds of the video. Each input example represents a video and contains a vector of size 1152 i.e. concatenation of 1024 size visual feature and 128 size audio vector. 

The other data given for the challenge is frame-level data, which contains array of 1152 size feature vectors of 300 frame corresponding to 300 seconds of video. 

For the competition we used only video level features.

You can downloda the partial data using following commands

```
# Video-level
mkdir -p ~/yt8m/v2/video
cd ~/yt8m/v2/video
curl data.yt8m.org/download.py | shard=1,100 partition=2/video/train mirror=us python
curl data.yt8m.org/download.py | shard=1,100 partition=2/video/validate mirror=us python
curl data.yt8m.org/download.py | shard=1,100 partition=2/video/test mirror=us python
```

## Download complete video level data

To downloda complete data, use following command:

```
curl data.yt8m.org/download.py | partition=2/video/train mirror=us python
```

Please refer to [challenge website](https://research.google.com/youtube8m/download.html) for more options of downloading.


## Train the model:

Please use the following command to train video classification model from scratch. Note that here we have used *model* as *NeuralNetworkModel*. You can also specify *BranchedNNModel* or *CNNModel*. By default the model used is basic Logistic Regression. For more info on command line arguments, please refer to *train.py* file.

```
time CUDA_VISIBLE_DEVICES=1 python train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --train_data_pattern=~/data_youtube8m/train_data/train*.tfrecord --train_dir ~/checkpoint_dir --start_new_model --num_epochs=20 --base_learning_rate=0.001 --model=NeuralNetworkModel
```

## Evaluate the model:

To evaluate the model, please following command. Refer to original [git repo](https://github.com/google/youtube-8m#evaluation-and-inference) for more info.

```
time CUDA_VISIBLE_DEVICES=1 python eval.py --eval_data_pattern=~/data_youtube8m/validation_data/validate*.tfrecord --train_dir ~/checkpoint_dir --run_once
```

## Test the model:

To generate the output file and the model zip file, to upload on the Kaggle, please use the following command.

```
time CUDA_VISIBLE_DEVICES=1 python inference.py --train_dir ~/checkpoint_dir  --output_file=kaggle_solution_sub5.csv --input_data_pattern=~/data_youtube8m/test_data/test*.tfrecord --output_model_tgz=my_model.tgz
```

For more information, please refer to [Kaggle Challenge website](https://www.kaggle.com/c/youtube8m-2018) or [YouTube dataset page](https://research.google.com/youtube8m/index.html) or [Google Git Repo](https://github.com/google/youtube-8m).



