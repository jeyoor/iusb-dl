## Applied Deep Learning - Lecture for 2019-09-18

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

