# Applied Deep Learning - Lecture for 2019-09-18

## Answers to in-class questions

- how do you get your editor to show code hints from python docstrings?
   - in [emacs](https://www.emacswiki.org/emacs/PythonProgrammingInEmacs)
   - in [vim](https://realpython.com/vim-and-python-a-match-made-in-heaven/)
   - in [pycharm](https://www.jetbrains.com/help/pycharm/documenting-source-code.html)
- where can we find a graph of the softmax function?
   - from [dataaspirant.com](https://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/)
- what is the purpose and function of the "moment" parameters (beta1 and beta2) in the Adam optimization algorithm?
   - from [machinelearningmastery.com](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
- what tools are available for visualizing the layers in a neural network?
   - [TensorBoard's Graph Visualization Tool](https://www.tensorflow.org/tensorboard/r1/graphs) draws diagrams for tensorflow models (not sure how to get this to work with Keras)
- what tools are available for visualizing saliency maps for a neural network?
   - [keras-vis](https://github.com/raghakot/keras-vis) is a popular library for creating saliency maps
- what tools are available for visualizing connections between neural network states and input data with a "brain-like" diagram?
   - [t-SNE](https://lvdmaaten.github.io/tsne/) is a popular algorithm for this type of visualization
   - There's also a video of the 3D t-SNE representation of the MNIST handwritten digits dataset available [here](https://lvdmaaten.github.io/tsne/examples/mnist_tsne.mov)
- what is the meaning of the "interpolation" parameter to the ImageDataGenerator class in Keras?
   - [Keras's flow_from_directory docs](https://keras.io/preprocessing/image/#flow_from_directory) seem to indicate that this interpolation parameter is used when trying to adjust an image to fit the required input size
      - It looks like these resizing options are coming from the [PIL Image.resize routine](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize)
      - There's more explanation of these options in these videos
         - [nearest neighbor and bilinear interpolation](https://www.youtube.com/watch?v=AqscP7rc8_M)
         - [Bicubic interpolation](https://www.youtube.com/watch?v=poY_nGzEEWM)

##Introduction

This repo contains python notebooks and info for the Fall 2019 IUSB Deep Learning Class.

The data is from Kaggle's Cats and Dogs challenge.

https://www.kaggle.com/c/dogs-vs-cats

## Discussion Topics

- Reiterating the importance of training data
                Dogs/cats

> > - Keras's functional vs. sequential APIs
> > - Optimizers (ADAM, SGD)
> > - Learning rate

> > - Activation functions
> >    - final dense layer (sigmoid for binary classification vs. softmax for
> > choosing from multiple classes)

> >    - hidden layers (usually relu or some relu variant)

Discussing them what an HDF5 "file" is would be useful.
(It's a zip file, but instead of files, it holds data structures)

> > - data set naming (train, test, holdout VS train, valid, holdout VS train,
> > valid, test VS ...)
> >    - emphasize that mnist and dogs-cats don't have a holdout set
> >    - emphasize that different names are used in different contexts
> >    - emphasize that you have to look it up each time through

> > Discuss Overfitting and Local Minima/Maxima

Local Minima/Maxima
https://youtu.be/IHZwWFHWa-w?t=409

Overfitting discussion
https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/

