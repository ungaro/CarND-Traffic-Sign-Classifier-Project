# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[graphimage]: ./examples/graph.png "Graph"
[samplesimage]: ./examples/samples_per_class.png "Samples Per Class"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image_TP]: ./examples/image_transform_preprocess.png "Traffic Sign 1"
[PT1]: ./examples/PT1.png "Traffic Sign 1"
[PT2]: ./examples/PT2.png "Traffic Sign 1"
[PT3]: ./examples/PT3.png "Traffic Sign 1"
[PT4]: ./examples/PT4.png "Traffic Sign 1"
[foundimages]: ./examples/found_images.png "Found Images"
[tb1]: ./examples/tensorboard1.png "Tensorboard 1"
[tb2]: ./examples/tensorboard2.png "Tensorboard 2"
[tb3]: ./examples/tensorboard3.png "Tensorboard 3"
[conv1]: ./examples/featuremap.png "Features"
[conv2]: ./examples/conv1.png "Conv2D 1"
[conv3]: ./examples/conv2.png "Conv2D 2"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among different classes. 


![alt text][samplesimage]




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Pre-processing for image data:

1. Applied histogram equalization to remove brightness effect.
2. Scaled images between -.5 and .5, by dividing by 255. and subtracting .5.

Here is an example of a traffic sign image before and after histogram normalization.

![alt text][PT1]
![alt text][PT2]
![alt text][PT3]
![alt text][PT4]

I've decided to let the neural network decide which channel is the best for classifying the data.

To add more data to the the data set, I rotated, translated and sheared images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I've visualized my model in TensorGraph. In the notebook, you should set LOG = True to get tensorflow log data for Tensorboard.

![alt text][graphimage]
This model visualized here has 6 conv layers. 
Final model has 8. 

The first module in the model above is comprised of 3 1X1 filters. We didn't converted our images to grayscale during pre-processing, we will let nn to decide.

The next 3 layers are  32, 64, 128 and 256 3X3 filters followed by max-pooling and dropouts. The output from each of the convolution module is fed into a feedfoward layer. Fully connected layer has access to outputs from low level and higher level filters and can choose the features that works the best. The FF layers are 2 hidden layers with 1024 neurons in each layer. Additional dropout layers are applied after each of the fully connected layers.

The idea of using drop outs heavily is to avoid overfitting and force the network to learn multiple models for the same data. 

After the fully connected layers softmax layer is used to compute the log-loss of model prediction. In addition a l2- regularization cost is included to penalize large model weights.



My final model consisted of the following layers:

| Layer         							|     Description	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Features         					  1X1x3 | 32x32x3 RGB image   							| 
| Convolution + RELU 				 5x5x32 | 1x1 stride, same padding				 		|
| Convolution + RELU 				 5x5x32 | 1x1 stride, same padding				 		|
| Max-Pooling + DropOut	(x)			 		| 3x3 max-pooling, 0.5 Dropout					|
| Convolution + RELU 				 5x5x64 | 1x1 stride, same padding				 		|
| Convolution + RELU 	 			 5x5x64	| 1x1 stride, same padding				 		|
| Max-Pooling + DropOut	(x)			 		| 3x3 max-pooling, 0.5 Dropout					|
| Convolution + RELU 	 		    5x5x128	| 1x1 stride, same padding					 	|
| Convolution + RELU 	 		    5x5x128	| 1x1 stride, same padding						|
| Max-Pooling + DropOut	(x)			 		| 3x3 max-pooling, 0.5 Dropout					|
| Convolution + RELU 	 		    5x5x256	| 1x1 stride, same padding					 	|
| Convolution + RELU 	 		    5x5x256	| 1x1 stride, same padding						|
| Max-Pooling + DropOut	(x)			 		| 3x3 max-pooling, 0.5 Dropout					|
| Flatten				 					| Flatten E(x)									|
| Fully Connected 					1024	| 												|
| DropOut				 					|  0.5 Dropout									|
| Fully Connected 					1024	| 												|
| DropOut 	 								|  0.5 Dropout 									|
| --------------	   						| --------------      							|
| Softmax									|												|
| Prediction								| 	        									|
 
The Shapes of each layer after activation is as follows:
##### Shapes of Layers
- Features.: (?, 32, 32, 3)
- Conv 0.: (?, 32, 32, 3)
- Conv. 1: (?, 32, 32, 32)
- Conv. 2: (?, 16, 16, 32)
- Conv. 2 after Dropout: (?, 16, 16, 32)
- Conv. 3: (?, 16, 16, 64)
- Conv. 4: (?, 8, 8, 64)
- Conv. 4 after Dropout: (?, 8, 8, 64)
- Conv. 5: (?, 8, 8, 128)
- Conv. 6: (?, 4, 4, 128)
- Conv. 6 after Dropout: (?, 4, 4, 128)
- Conv. 7: (?, 4, 4, 256)
- Conv. 8: (?, 2, 2, 256)
- Conv. 8 after Dropout: (?, 2, 2, 256)
- 
- Flattened Conv. 2.: (?, 8192)
- Flattened Conv. 4.: (?, 4096)
- Flattened Conv. 6.: (?, 2048)
- Flattened Conv. 8.: (?, 1024)
- 
- Sum of Flattened Conv. Layers.: (?, 15360)
- 
- Fully Connected 1: (?, 1024)
- Fully Connected 1 Droput: (?, 1024)
- Fully Connected 2: (?, 1024)
- Fully Connected 2 Droput: (?, 1024)
- Fully Connected 3: (?, 43)





#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Augmentation starts high at the start so the model learns overall features of traffic signs, and gradually will be reduced to fine tune the model. 

Training steps are:

Generate 10 new images per image in the training set using data augmentation
Split data into training and validation sets such that the validation set is 25% of the training set.
After first 10 epochs, lower the augmentation by a factor of 0.9 per epoch.


tf.nn.conv2d function can be used to build a convolutional layer which takes these inputs:

 width x height x num_channels = (activation) from the previous layer( 4-D tensor),  [n x width x height x num_channels]

filter= trainable variables defining the filter. We start with a random normal distribution and learn these weights. It’s a 4D tensor whose specific shape is predefined as part of network design. If your filter is of size filter_size and input fed has num_input_channels and you have num_filters filters in your current layer, then filter will have following shape:

[filter_size filter_size num_input_channels num_filters]

##### strides
Defines how much you move your filter when doing convolution. In this function, it needs to be a Tensor of size>=4 i.e. [batch_stride , x_stride , y_stride , depth_stride]. 

- batch_stride is always 1 as you don’t want to skip images in your batch. 
- x_stride and y_stride are same mostly and the choice is part of network design and we shall use them as 1 in our example. 
- depth_stride is always set as 1 as you don’t skip along the depth.

##### padding
SAME means we shall 0 pad the input such a way that output x,y dimensions are same as that of input.

After convolution, we add the biases of that neuron, which are also learnable/trainable. Again we start with random normal distribution and learn these values during training.

Now, we apply max-pooling using tf.nn.max_pool function that has a very similar signature as that of conv2d function.


tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

We are using k_size/filter_size as 2*2 and stride of 2 in both x and y direction. If you use the formula (w2= (w1-f)/S +1; h2=(h1-f)/S +1 ) mentioned earlier we can see that output is exactly half of input. These are most commonly used values for max pooling.

Finally we use a RELU function.


##### Hyperparameters
I chose a learning rate of 1e-3( 0.001), batch size of 256 and a L-2 regularization on weights of \( 10^{-5} \) to avoid overfitting.

I trained the model for 15000 batches in the first, where i need improvement in 5000 batches. 

##### Stopping conditions
I used accuracy of validation data as a criteria to monitor if model was overfitting. I stopped training if the validation score didnt improve for 1000 consecutive iterations.

##### Optimization
I used adamoptimizer with default settings for optimization. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


I wanted to do some custom graph and these experiments ended with 1 1X1x3 conv. network followed by four times two conv. networks with filter sizes 32,64,128 and 256 + Dropout and summing features from them. There are two Fully Connected Layers followed by Softmax and Prediction.

First i've experimented with 2 Conv layers, and these experiments folded into 3 and 4 combinations.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 99.8% 
* test set accuracy of 97.2%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][foundimages]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        					|     Prediction	        				| 
|:-----------------------------------------:|:-----------------------------------------:| 
| Right-of-way at the next intersection     | Right-of-way at the next intersection   	| 
| Turn right ahead     						| Turn right ahead 							|
| Speed limit (30km/h)						| Speed limit (30km/h)						|
| No entry	      							| No entry					 				|
| Bumpy road								| Bumpy road      							|
| Speed limit (120km/h)						| Speed limit (120km/h)     				|




The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.




![alt text][foundimages]


1- Top 5 Predictions:
- 'Right-of-way at the next intersection' with probability: 0.0608
- 'Beware of ice/snow' with probability: 0.0226
- 'Dangerous curve to the right' with probability: 0.0224
- 'General caution' with probability: 0.0224
- 'Traffic signals' with probability: 0.0224

2- Top 5 Predictions:
- 'Turn right ahead' with probability: 0.0604
- 'Keep left' with probability: 0.029
- 'Go straight or left' with probability: 0.0222
- 'Roundabout mandatory' with probability: 0.0222
- 'Ahead only' with probability: 0.0222

3- Top 5 Predictions:
- 'Speed limit (30km/h)' with probability: 0.0596
- 'Speed limit (70km/h)' with probability: 0.0286
- 'Speed limit (50km/h)' with probability: 0.0261
- 'Speed limit (80km/h)' with probability: 0.0232
- 'Speed limit (120km/h)' with probability: 0.0231

4- Top 5 Predictions:
- 'No entry' with probability: 0.0608
- 'Stop' with probability: 0.0224
- 'Priority road' with probability: 0.0224
- 'Bumpy road' with probability: 0.0224
- 'Wild animals crossing' with probability: 0.0224

5- Top 5 Predictions:
- 'Bumpy road' with probability: 0.0602
- 'Road work' with probability: 0.0266
- 'Traffic signals' with probability: 0.0243
- 'Road narrows on the right' with probability: 0.0237
- 'Bicycles crossing' with probability: 0.023

6- Top 5 Predictions:
- 'Speed limit (120km/h)' with probability: 0.0608
- 'Speed limit (100km/h)' with probability: 0.0226
- 'Speed limit (70km/h)' with probability: 0.0224
- 'Speed limit (50km/h)' with probability: 0.0224
- 'Speed limit (80km/h)' with probability: 0.0224


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below are some images from tensorboard. Accuracy, Loss graphs...
![alt text][tb1]


You can see in the second image we've added a log for our features in image section.
![alt text][tb2]


 and we can also see the graph elements by clicking their respective group.
![alt text][tb3]

 and we can also see the graph elements by clicking their respective group.


Here are features filtered by 2 conv. networks.
Features
![alt text][conv1]

Conv1
![alt text][conv2]

Conv2
![alt text][conv3]