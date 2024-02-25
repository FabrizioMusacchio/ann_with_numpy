# Building a neural network from scratch using NumPy

Ever thought about building you own neural network from scratch by simply using NumPy? In this code example, we will do exactly that. We will build a simple feedforward neural network and train it on the MNIST dataset. The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a collection of 28x28 pixel grayscale images of handwritten digits (0-9). It is a popular dataset for getting started with machine learning and computer vision. The dataset contains 60,000 training images and 10,000 test images. The goal is to train a model that can correctly classify the images into their respective digit classes.

The entire tutorial can be found in this [blog post](https://www.fabriziomusacchio.com/blog/2024-02-25-ann_from_scratch_using_numpy/).

For reproducibility:

```bash
conda create -n numpy_ann python=3.11
conda activate numpy_ann
conda install -y mamba
mamba install -y numpy matplotlib keras ipykernel
```

If you want to run the code on an Apple Silicon chip, follow [these instructions](https://www.fabriziomusacchio.com/blog/2022-11-10-apple_silicon_and_tensorflow/) to install TensorFlow (required by Keras).



