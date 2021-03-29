# **Traffic Sign Recognition** 

## Project Outline

The goal of this project is to develop a model for traffic sign recognition. In this project, we will use the German Traffic Sign Dataset to explore, visualize, summarize, and process the dataset. Subsequently, we will design, train and test the deep learning model architecture, and use the model to make predictions on new images. Finally, we will analyze the softmax probabilities of the new images, and summarize the results obtained from our model.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### 1. Project files

You're reading it! and here is a link to my [project code](/traffic-sign-classifier/). This project includes the Jupyter Notebook, the report.html file which is the exported HTML version of the notebook, and the directory test_images for the test images found on the web.

### 2. Data Set Summary & Exploration

**Summary of Dataset.**

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

For an exploratory visualization of the dataset, see cells 3 (color) and 6 (grayscale) for visualization of the dataset.

### 3. Design and Test a Model Architecture

**Preprocessed the traffic sign.**

For our pre-processing of the dataset, we chose to grayscale the images since traffic signs are different in their shapes and contents. Although color of the images will indeed help us in classifying our traffic signs, but we can rely on the different shapes and contents in the signs for this context. As such, the preprocessing of the dataset includes grayscaling the images, and then normalizing them from [0, 255] to [0, 1].

**Model architecture**

Refer to cell 51 for the full model architecture, in function LeNet.

For the model architecture, I implemented an adapted fversion of the LeNet architecture. The model has 3 convolutional layers with 2 pooling layers in between them, followed by one flatten layer, and 3 fully connected layers with 2 dropout layers in between them.

My final model consisted of the following layers:

| Layer         		|     Description	        					            | 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					            | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	            |
| RELU					|												            |
| Max Pooling           | 2x2 ksize, 2x2 strides, valid padding, outputs 14x14x6    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	            |
| RELU					|												            |
| Max Pooling           | 2x2 ksize, 2x2 strides, valid padding, outputs 5x5x16     |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x412 	            |
| Flatten           	| output = 412                                              |
| Fully connected		| output = 122            									|
| RELU  				|                       									|
| Dropout 				| keep probability at 0.5									|
| Fully connected		| output = 84            									|
| RELU  				|                       									|
| Dropout 				| keep probability at 0.5									|
| Fully connected		| output = 43            									|
|                       |                                                           |

**Model training process**

Refer to cells 52, 53 and 55 for the training of model process. To train the model, I used:
* Learning rate of 0.00097
* Adam optimization algorithm for stochastic gradient descent as the optimizer
* cross-entropy as the loss function
* Softmax activation function for our output layer
* Batch size of 156
* Epochs set at 25
* Dropout probability set at 0.5 for every dropout layer

**Approach taken for getting validation set accuracy of at least 0.93**

Refer to cells 55, 56 and 95 for the full pipeline to evaluate the model.

My final model results were:
* training set accuracy of 0.984.
* validation set accuracy of 0.943.
* test set accuracy of 0.909.

My first approach was to use the LeNet architecture. Since the LeNet architecture has a great performance with recognizing handwritings, implementing it on classification of traffic signs would be make sense.

Subsequently, I tested the LeNet model and obtained a mediocre training accuracy of around 90%. I implemented another convolutional layer to increase the depth of the feature maps, which ultimately led to an output layer of 1x1x412.

This increased the training accuracy by abit more, but I felt that I could increase the accuracy of the model by preventing overfitting from happening. Thus, I added dropout layers in between every fully connected layers. In addition, I increased the batch size to 156, and increased the number of epochs as well.

This led to the model to become more accurate, and I managed to achieve the above validation set accuracy of 0.943 during the training process. To better visualize the improvements in the model accuracy after every epoch during the trianing process, you may refer to cell 56 for the graph of Test Accuracy vs Number of Epochs, and the graph of Validation Accuracy vs Number of Epochs.

### Test a Model on New Images

**1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.**

For the five German traffic signs that I found on the web, you may refer to the test image folder at [**self-driving-project/traffic-sign-classifier/test_images/**](/traffic-sign-classifier/test_images/). You may also refer to cells 143 and 144 for visualization of the images (color and grayscale).

For the images chosen, signs like the General Caution and the Bumpy Road signs are highly similar in shapes. We test to see the quality of our prediction model in predicting correctly signs are that highly similar.

**2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.**

Refer to cells 145 and 147 for detailed predictions on the new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      	| General caution 		    				    | 
| Speed limit (30km/h)  | Speed limit (30km/h)				            |
| Go straight or left	| Go straight or left			                |
| Ahead only	      	| Ahead only					    			|
| Bumpy road			| Bumpy road     					    		|
| 			            |               					    		|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of which was at 94.3%. This shows that our model was not overfitted. However, to properly test our model, we should definitely use a larger test dataset in the future, to fully test for overfitting/underfitting.

**3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.**

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

* For the first image, the model is very sure that this is a stop sign (probability of 1.0)
* For the second image, the model is relatively sure that this is a stop sign (probability of 0.71)
* For the third image, the model is almost 100% sure that this is a stop sign (probability of 0.99)
* For the fourth image, the model is very sure that this is a stop sign (probability of 1.0)
* For the fifth image, the model is relatively sure that this is a stop sign (probability of 0.60)
