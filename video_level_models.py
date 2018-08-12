# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


class NeuralNetworkModel(models.BaseModel):
    """It is simple 2 layer neural network with L2 regularization and relu activation"""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """
        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""
        print("Model used: Simple 2 layer neural network")
        # h1 size = 1152 (input vector size) x 576 (half of input size)
        h1 = slim.fully_connected(model_input, int(model_input.get_shape().as_list()[-1] / 2), activation_fn=tf.nn.relu,
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))

        # output size = 576 (half of input size) x 3862 (num of classes)
        output = slim.fully_connected(h1, vocab_size, activation_fn=tf.nn.sigmoid,
                                      weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}


class BranchedNNModel(models.BaseModel):
    """Branched 'v' shaped neural network model with L2 regularization."""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """
        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        print("Model used: Branched 'V' neural network")
        # separate the input vector of size 1152 into video and audio part of size 1024 and 128 length respectively
        video,audio = tf.split(model_input,[1024,128],axis=1)

        # dimensionality reduction

        # a dense layer to reduce the dimensions of video; output size=512
        vdNN = slim.fully_connected(video, 512, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

        # a dense layer to reduce the dimensions of video; output size=64
        adNN = slim.fully_connected(audio, 128, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

        # concatenate; output size=576
        mix = tf.concat([vdNN,adNN],axis=1)

        # dense layer; output size=288
        h1 = slim.fully_connected(mix, int(mix.get_shape().as_list()[-1]/2), activation_fn=tf.nn.relu,
                                  weights_regularizer=slim.l2_regularizer(l2_penalty))

        # final softmax layer for classification; output size 3862 (=number of classes)
        output = slim.fully_connected(h1, vocab_size, activation_fn=tf.nn.sigmoid,
                                      weights_regularizer=slim.l2_regularizer(l2_penalty))

        print(output)
        return {"predictions": output}


class CNNModel(models.BaseModel):
    """Convolutional neural network model with structure:
       videoNN      audioNN
             \     /
             matmul
               |
             CNN1
               |
             avg_pool
               |
             CNN2
               |
             flatten
               |
         output softmax
    """

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """Creates a logistic model.

        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        # separate the input vector of size 1152 into video and audio part of size 1024 and 128 length respectively
        print("Model used: CNN model")
        video,audio = tf.split(model_input,[1024,128],axis=1)

        # dimensionality reduction
        # a dense layer to reduce the dimensions of video
        vdNN = slim.fully_connected(video, 32, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
        adNN = slim.fully_connected(audio, 32, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

        # calculate outer product of video and audio
        # adds extra dimension to video vector; output size=32x1
        vd = tf.expand_dims(vdNN,-1)

        # adds extra dimension to audio vector; output size=32x1
        ad = tf.expand_dims(adNN,-1)

        # convert the column audio vector to row vector; output size=1x32
        ad = tf.transpose(ad,perm=[0,2,1])

        # calculate outer product; output size=32x32
        mix = tf.matmul(vd,ad)

        # add extra dimension to make matrix 3D so that CNN can be applied; output size=32x32x1
        mix = tf.expand_dims(mix,-1)

        # first convolutional layer; output size=30x30x8
        conv1 = tf.layers.conv2d(inputs=mix, filters=8, kernel_size=[3,3])

        # average pooling; output size=15x15x8
        avgpool = tf.layers.average_pooling2d(inputs=conv1, pool_size=2,strides=2)

        # second convolutional layer; output size=13x13x4
        conv2 = tf.layers.conv2d(inputs=avgpool, filters=4, kernel_size=[3,3])

        # flatten the output; output size=676
        flat = tf.contrib.layers.flatten(inputs=conv2)

        # final softmax layer for classification; output size 3862 (=number of classes)
        output = slim.fully_connected(flat, vocab_size, activation_fn=tf.nn.sigmoid,
                                      weights_regularizer=slim.l2_regularizer(l2_penalty))

        print(output)
        return {"predictions": output}


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
