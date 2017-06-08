# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. The link to my github repository is [here](https://github.com/urs-waldmann/CarND-Behavioral-Cloning-P3).

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral_cloning.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* behavioral_cloning.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the first (easy) track by executing 
```sh
python drive.py behavioral_cloning.h5
```

#### 3. Submission code is usable and readable

The behavioral_cloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the [convolutional neural network from NVIDIA](https://arxiv.org/pdf/1604.07316.pdf). Thus my model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (behavioral_cloning.py lines 40-56) 

The model includes RELU layers to introduce nonlinearity (e.g. code line 43), and the data is normalized in the model using a Keras lambda layer (code line 42). 

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers. I tried to add dropout layers after the first three fully connected layers (code lines 50, 52 & 54). This did not improve overfitting. I also tried with less convolutional layers (by commenting out code line 47 & 49; together and each alone) but also this did not improve overfitting. To avoid overfitting instead I increased my training data. For track 1 I was using 28 732 training images and 7 184 validation images.
When successfully done with track 1, I quickly tried track 2. Therefore I recorded more data. My training data contained the old training data from track 1 plus the newly recorded data and thus contained 79 074 in total. With this amount of data I had even less overfitting problems. Thus my assumption is that I should have used even more training data for track 1 to improve the little remaining overfitting even more. Another option could have been, I assume, to use dropout with an even more radical ratio to drop activations. I used 75% which did not improve overfitting, but I guess with 90% it would have improved overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 14-31). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (behavioral_cloning.py line 60).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and focusing on driving smoothly around curves. I used data where I drove both clock-wise and counterclock-wise. I aslo included few data from track 2 to generalize the neural network.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try several known model architectures with a small training data set, choose a model architecture, augment the training data set and modify the model architecture to avoid overfitting.

My first step was to use the [LeNet](http://yann.lecun.com/exdb/lenet/) convolution neural network model. I thought this model might be appropriate because it is a known model architecture that works very well on the ImageNet data set. This model worked well  until I heavily augmented the training data set.

My second step was to try the [convolutional neural network from NVIDIA](https://arxiv.org/pdf/1604.07316.pdf). I thought this is even more appropriate since NVIDIA is using this model architecture to drive autonomous cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (behavioral_cloning.py lines 40-56) consisted of a convolution neural network with the following layers and layer sizes:


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 53 874 number of data points. I then preprocessed this data by normalizing and mean centering it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a contiuously decreasing training and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
