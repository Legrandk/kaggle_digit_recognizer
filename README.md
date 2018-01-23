# Digit Recognizer

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

[Kaggle Competition Link](https://www.kaggle.com/c/digit-recognizer)



## Datasets
The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

* [Train](https://www.kaggle.com/c/digit-recognizer/download/train.csv)
* [Test](https://www.kaggle.com/c/digit-recognizer/download/test.csv)



## Models

### LeNet5 V1

My main goal is to find a good architecture to work with and be able to deal with Bias/Variance problem very fast. I choosen to start with LeNet-5 because it was the most simple and runs pretty fast on my computer. I've also tested AlexNet but it takes to much time to train in my computer unfortunatelly.

From a vanila LeNet-5 I made the following adaptions to start working:
 1. Use ReLU activation function instead sigmod.
 1. Filters use SAME padding.
 1. Use Max Pooling instead Average Pooling.
 1. Use a softmax layer as output.

This first version performed pretty well at the first run using the following hyperparmeters:
 1. Epochs: 50
 1. Optimizer: Adam with default values (LR=0.001)
 1. Batch Size: 32
 
This was the Kaggle scoring and positioning the model got:
* K-Scoring: 0.98857
* K-Position: 760
* Metrics:
 * Train/F1_Score: 0.99985125633534111
 * Train/Acc: 0.99630952
 * Validation/F1_Score: 0.9990811288857685
 * Validation/Acc: 0.98773807
 

Then I decided to try improve the model just increasing the number of epochs up to 165. It performed better and got the following Kaggle scoring:

* K-Scoring: 0.98885
* K-Position: 751
* Metrics:
 * Train/F1_Score: 1.0
 * Train/Acc: 1.0
 * Validation/F1_Score: 0.99947006736346533
 * Validation/Acc: 0.99000001



### LeNet5 V2

In the previous model I've to solve the Variance problem because I've overfit the Train model. I've tried different (1) regularization techniques as L2 Regularization and Dropout, and (2) other architectures. The best performace I got was using Dropout.

The new model LetNet V2 has two Dropout layers one just before the flatten layer and the last just after the first FC layer using a keep probability of 0.8.

While exploring other architectuctures I found the following changes helps dealing with Variance:
1. Increase the number of filters in the first convolution layer up to 30 filters.
1. Decrease the filter size used in the second convolution layer. Now it uses 3x3 filter size.

Concerning the hyperparameters values I made the following changes:
* Epochs: 200
* Barch Size: 200

This model performed much better than V1 and got the following Kaggle Scoring:

* K-Scoring: 0.99042
* K-Position: 652 (Top 40%)
* Architecture:
 * Base: LetNet5 V1
 * Conv1: Number of filters=30
 * Conv2: Filter size: 3x3
 * Dropout: 0.2
* Metrics:
 * Train/F1_Score: 1.0
 * Train/Acc: 1.0
 * Validation/F1_Score: 0.99966877260234099
 * Validation/Acc: 0.99297619
 


### LeNet5 V3

In this version I'd like to try solving the Variance problem gathering more data which will helps the model to work better with cases it hasn't seen before.

Kaggle Train set has only 42.000 samples and its Test set 28.000. So I decided to try working directly with the original [MNIST](http://yann.lecun.com/exdb/mnist/) sets. MNIST sets are splitted as 60.000 samples for training and validation and 10.000 samples for test.

I got 42% more training samples for training my model, using 55.000 for training and 5.000 for validation. 

Using this data the model improved drastically and got the following Kaggle scoring:

* K-Scoring: 0.99900
* K-Position: 30 (Top 2%)
* Architecture:
 * Base: LetNet5 V2
 * Train/Test sets: Full MNIST dataset
* Metrics:
 * Train/F1_Score: 1.0
 * Train/Acc: 1.0
 * Validation/F1_Score: 0.99966816871350095
 * Validation/Acc: 0.99360001



## Benchmarks Models

| Model | K-Scoring | K-Position | F1 Score | Acc train/val | Loss Acc/val | Epochs | Batch Size | Seed |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| LeNet5 v3  | 0.99900 | 30  | 1.0 / 0.99966816871350095  | 1.0 / 0.99360001 | 1.3700266e-05 / 0.040703569 | 200  | 200 | 1 |
| LeNet5 v2  | 0.99042 | 652 | 1.0 / 0.99966877260234099  | 1.0 / 0.99297619 | 5.616491e-05 / 0.054907657 | 200  | 200 | 1 |
| LeNet5 v1  | 0.98885 | 751 | 1.0 / 0.99947006736346533  | 1.0 / 0.99000001 | 0.0 / 0.19570193 165 |  32 | 1 |
| LeNet5 v1  | 0.98857 | 760 | 0.99985125633534111 / 0.9990811288857685  | 0.99630952 / 0.98773807 | 0.012848322 / 0.093898751 | 50  |  32 | 123 |
| AlexNet v1 | 0.98714 | -   | N/A / N/A  | 0.9946726  / 0.98952383 | 0.02303471   / 0.045152489 | 10   |  32 | 123 |


## License
The MIT License. Copyright 2018 (c) Alex Martin
