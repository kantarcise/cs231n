Part of [CS231n Winter 2016](../index.md)

---
# Lecture 5:  Training Neural Networks,  Part I

![5001](../img/cs231n/winter2016/5001.png)

Some things to think about project proposals.

![5002](../img/cs231n/winter2016/5002.png)

![5003](../img/cs231n/winter2016/5003.png)

You can finetune!

![5004](../img/cs231n/winter2016/5004.png)

Chop of the classifier, and train the neural network as a feature extractor? How man ?

![5005](../img/cs231n/winter2016/5005.png)

You have a lot of pretrained datasets!

![5006](../img/cs231n/winter2016/5006.png)

LOL. During hyper parameter optimization, you will need a lot of compute. Be careful.

![5007](../img/cs231n/winter2016/5007.png)

Finite compute.

![5008](../img/cs231n/winter2016/5008.png)

We are here. 
## batch a data - forward - backprop - update

![5009](../img/cs231n/winter2016/5009.png)

We have a Optimization problem at hand.

![5010](../img/cs231n/winter2016/5010.png)

NN's can get really large.

![5011](../img/cs231n/winter2016/5011.png)

We need calculus - just chain rule.

![5011](../img/cs231n/winter2016/5011.png)

Chain rule seen here.

![5012](../img/cs231n/winter2016/5012.png)

We need a forward/backward API.

![5013](../img/cs231n/winter2016/5013.png)

Simple Multiplication Gate:

![5014](../img/cs231n/winter2016/5014.png)

LEGO's.

![5015](../img/cs231n/winter2016/5015.png)

We saw the activation functions here, first time ever.

![5016](../img/cs231n/winter2016/5016.png)

The resemblance of NN and Brain.

![5017](../img/cs231n/winter2016/5017.png)

FC are only the layers that are not input (which have computation).

![5018](../img/cs231n/winter2016/5018.png)

## History - Zoomed Out Field!

![5019](../img/cs231n/winter2016/5019.png)

1960 is the start, the perceptron.
## the perceptron

they had to actually build from circuits and electronics.

Activation function was a binary step function, it was not differentiable. 

So back propagation was not even here. It came much later. 

There was no loss function also.

![5020](../img/cs231n/winter2016/5020.png)

They started to stack perceptons, still building on hardware. Just rules.

![5021](../img/cs231n/winter2016/5021.png)

Scientist was excited but would under deliver, they did not work good.

First time you see back propagation like rules, seen on paper. People got excited again. Training would get stuck and it would not work.

![5022](../img/cs231n/winter2016/5022.png)

It was 2006, when a 10 layer NN would actually trains properly.

Which is based on RBZ:
# RBZ - Restricted Boltzman Machine.

Instead of training all 10 layers all at once, they came up with this unsupervised pretraining scheme. You train your first layer using an unsupervised objective than you train your second layer on top of it and third and fourth.

Once all of them are trained, than you start backpropogation. You start the fine-tuning step.

You actually can start with backpropogation for training but you have to be very careful with initialization.

They used sigmoids and sigmoids are really not good activation functions.

![5023](../img/cs231n/winter2016/5023.png)

Things started to work relatively well.

In 2010, strong results. 

In speech recognition area, they had this `GMM` - `HMM` stuff, they swapped out one part and subbed in a NN, THAT GAVE THEM HUGE IMPROVEMENTS.

In Vision - 2012 crushed the competition. Field exploded.

![5024](../img/cs231n/winter2016/5024.png)

Why this happened in 2012? 
- We figured out initialization
- We had better activations
- We had more data
- We had better compute.

---

Next 2 lectures will be about, these:

![5025](../img/cs231n/winter2016/5025.png)
## activation functions:

![5026](../img/cs231n/winter2016/5026.png)

Different proposals:

![5027](../img/cs231n/winter2016/5027.png)

Historically, squashes a number between 0 and 1.

![5028](../img/cs231n/winter2016/5028.png)

Problems (Vanishing gradient):

![5029](../img/cs231n/winter2016/5029.png)

We would like to back propagate on sigmoid gate,

![5030](../img/cs231n/winter2016/5030.png)

Gradient is very low when X is 10 or $-10$
# Slope is actually 0. You can see that.

## If your neuron is saturated (which means they are either 0 or 1) than gradient cannot backpropogate through the network.

Only active region of the sigmoid will work.

![5031](../img/cs231n/winter2016/5031.png)

When you preprocess data, you want to make sure that it is zero centered. 

![5031](../img/cs231n/winter2016/5031.png)
## If your x's are all positive what happens on gradients on w ?

![5032](../img/cs231n/winter2016/5032.png)

All gradients of w are either all positive or all negative. 
$$X * grad(f)$$

If the gradient of the output of the neuron is positive. Then all your W gradients will be positive and vice versa.

The problem is you are constrained in the update you can make.

![5033](../img/cs231n/winter2016/5033.png)

## You can see this empirically, when you train with things that are not zero centered, you will observe slower convergence.

# You want to have things that are zero centered.

![5034](../img/cs231n/winter2016/5034.png)

Literally simply, just expensive.
## when we are training CNN's most of compute time is actually in convolutions and dot products

exp is however expensive.

Yann Lecun recommended using tanh() instead of sigmoids.

![5035](../img/cs231n/winter2016/5035.png)

Yeah, still vanishing gradients.
# In 2012 first convolutional neural networks paper, ReLu makes CNN's converge faster. 🎱

Just thresholding. This is the default.

![5036](../img/cs231n/winter2016/5036.png)
## Problems, not zero centered. when x < 0 gradient dies.

![5037](../img/cs231n/winter2016/5037.png)

If it is positive, just passes through.
# At x = 0 gradient is undefined.

## Whenever we are talking gradient in this course we are talking about sub-gradient, generalization of gradient to functions that are sometimes non differentiable.

![5038](../img/cs231n/winter2016/5038.png)
## In practice, if you are unlucky in your initialization, ded relus. They will not come back.

![5039](../img/cs231n/winter2016/5039.png)

Sometimes when you are training, you stop and look at the activation on neurons,
## as much as %10 or %20 of your network is ded. These are neurons that are never turned on for anything in the training data.

## To avoid this problem, initialize with instead of 0 bias, 0.01 bias

![5040](../img/cs231n/winter2016/5040.png)

An approach to solve ReLu. Neurons are not dying.
##  Andrej is not completely sold on them.

![5041](../img/cs231n/winter2016/5041.png)

Also PReLu is possible. The slope can be a parameter.

![5042](../img/cs231n/winter2016/5042.png)

2 months ago at the lecture times, a new paper came out:

![5043](../img/cs231n/winter2016/5043.png)

`Ian Goodfellow et al` is how you read the top left.

A different form of Neuron.
## It has 2 weights. 🤔

![5044](../img/cs231n/winter2016/5044.png)
## a lot of the optimization process is not just about loss function, it is more about the dynamics of the backwards flow of the gradients.

![5045](../img/cs231n/winter2016/5045.png)
## Sigmoids are used in LSTM and RNN's but there are reason for that.

![5046](../img/cs231n/winter2016/5046.png)
## Very common to zero center your the data = Along every single feature you subtract the mean.

### In ML literature (normalize the data) you do standardizing (normalize the standart deviation), or you can make sure the min and the max are within -1 and 1.

In images normalizing is not common.

![5047](../img/cs231n/winter2016/5047.png)

In Machine Learning these common but in images we are not using these:

- PCA - you can make the covariance structure become diagonal
- Whitened data - squish the data - covariance matrix becomes diagonal.
## PCA is not really common becuase images are really high dimentional objects with a lot of pixels so covariance matrices would be huge.

![5048](../img/cs231n/winter2016/5048.png)

In images, mean centering:

- AlexNet has the mean image of orange blob
- You can subtract based on RGB.

![5049](../img/cs231n/winter2016/5049.png)

People were not careful enough with this:

![5050](../img/cs231n/winter2016/5050.png)

How not to do it:

![5051](../img/cs231n/winter2016/5051.png)

If you set all weights to 0:

All output would be zero, there is nothing **symmetry breaking**.

![5052](../img/cs231n/winter2016/5052.png)

This is the first valid approach:

![5053](../img/cs231n/winter2016/5053.png)

In deep networks things go wrong with 0.01 w's, let's see an example:

```python
# sample a dataset first 1000 points 500 dimentional

# make hidden layers and nonlineararties

# use tanh

# take unit gaussian data and forward it in network

# in the first for loop, you are just forward propogating the network
```

What happens to the statistics of the neurons activations throughout the network with this initialization.
- We will look at mean and standard deviations
- we will plot histograms

![5054](../img/cs231n/winter2016/5054.png)

Means and standard deviations are printed, means stays around 0 but standard deviation plumbed down to 0.

![5055](../img/cs231n/winter2016/5055.png)

Why is this an issue?
## Think about the dynamics of the backward pass, to the gradients, when you have tiny numbers in the activations what do these gradients on these weights look like?

![5056](../img/cs231n/winter2016/5056.png)

Tiny X's will make the gradient really small.

When you multiply by W at every single layer, the gradients will just go to really small number.

What if we initialized with $1$ instead of $0.01$ ?

![5057](../img/cs231n/winter2016/5057.png)

Now we have overshot the other way LOL. Everything is completely saturated. All the neurons are either 1 or -1.

You die. You can train for very long time and you can see that loss is not moving at all. Because nothing is back propagating because all neurons are saturated and nothing is being updated.
# The Xavier Initialization!

Based on number input number, weight initialization changes.

- If you have a lots of input and you will end up with lower weights.
- If you have less inputs and your weights will be large.

In the notes there is text about how this is derived. This is reasonable.

But this paper  has the flaw of not taking into account tanh() 

![5058](../img/cs231n/winter2016/5058.png)

When we try with ReLU, it does not work at all.

![5059](../img/cs231n/winter2016/5059.png)

Someone pointed out, you are missing out an factor of 2. 

> ReLU neurons half your variance each time.

When you use this factor of 2, things build nicely.

![5060](../img/cs231n/winter2016/5060.png)

You have to be careful with initialization.

![5061](../img/cs231n/winter2016/5061.png)

This is the proper way. Xaiming He with factor of 2.

![5062](../img/cs231n/winter2016/5062.png)

There is also a data driven approach for initialization too.

---
## There is a technique that alleviates a lot of this problems.

# Batch Normalization

----
### Explained in assignment python notebook: ## 😍 

To understand the goal of batch normalization, it is important to first recognize that ==machine learning methods tend to perform better with input data consisting of uncorrelated features with zero mean and unit variance. ==

When training a neural network, we can preprocess the data before feeding it to the network to explicitly decorrelate its features. 

This will ensure that the first layer of the network sees data that follows a nice distribution. 

However, even if we preprocess the input data, the activations at deeper layers of the network will likely no longer be decorrelated and will no longer have zero mean or unit variance, since they are output from earlier layers in the network. 

Even worse, during the training process the distribution of features at each layer of the network will shift as the weights of each layer are updated.

The authors of BN paper hypothesize that the shifting distribution of features inside deep neural networks may make training deep networks more difficult. 

To overcome this problem, they propose to insert into the network layers that normalize batches. At training time, such a layer uses a mini-batch of data to estimate the mean and standard deviation of each feature. 

These estimated means and standard deviations are then used to center and normalize the features of the mini-batch. A running average of these means and standard deviations is kept during training, and at test time these running averages are used to center and normalize features.

It is possible that this normalization strategy could reduce the representational power of the network, since it may sometimes be optimal for certain layers to have features that are not zero-mean or unit variance. 

To this end, the batch normalization layer includes learnable shift and scale parameters for each feature dimension. 
  
----

This helps a lot. Basic idea is, you can make things unit Gaussian because it is a completely differentiable function. You can back propagate through it.

![5063](../img/cs231n/winter2016/5063.png)

N examples (mini batch) going through network, D features / activations of neurons, X makes sure every single column is normalized. 

By subtracting and dividing stuff.

![5064](../img/cs231n/winter2016/5064.png)
# Normally we had Fully Connected followed by non linearity (tanh - relu)

### Now we have BN layers right after FC layers.

## These BN's make sure everything is roughly unit Gaussian at every single step of the network (by normalizing them).

![5065](../img/cs231n/winter2016/5065.png)

A small patch on top of it. Allow the network to be able to shift and scale the bump.

Back propagation can take over and finetune over time.

![5066](../img/cs231n/winter2016/5066.png)
### Several nice properties of Batch Norm. 🥰

- Reduces the strong independence of initialization.
- Reduces the need for dropout and seems to actually help.

![5067](../img/cs231n/winter2016/5067.png)

At test time, you nu and sigma can be a running sum  in training and remember that in test time.

![5068](../img/cs231n/winter2016/5068.png)

It is good thing to use. But there is a runtime penalty.

----
### Tip: Layer Normalization 🐘

Batch normalization has proved to be effective in making networks easier to train, but the dependency on batch size makes it less useful in complex networks which ==have a cap on the input batch size due to hardware limitations==. 

Several alternatives to batch normalization have been proposed to mitigate this problem; one such technique is Layer Normalization. 

Instead of normalizing over the batch, we normalize over the features. In other words, when using Layer Normalization, each feature vector corresponding to a single data-point is normalized based on the sum of all terms within that feature vector.

![batchNorm layerNorm](../img/cs231n/winter2016/batchNorm_layerNorm.png)
### Simple Definitions

- **Batch Normalization** is normalizing the inputs to each batch of data by subtracting the mean and dividing by the standard deviation. This helps to stabilize the inputs to each layer, reduce the internal covariate shift, and enable faster and more stable training of deep neural networks. Batch normalization is typically applied to the inputs of each batch of data, hence the name "batch" normalization. It is often used in conjunction with other techniques such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

- **Layer Normalization** is normalizing the inputs across the features (or neurons) within a layer, rather than across the batch dimension. This helps to stabilize the inputs to each layer, reduce the internal covariate shift, and enable faster and more stable training of deep neural networks. Layer normalization is often used in recurrent neural networks (RNNs) and transformer models, where the inputs to each layer may have varying lengths and dimensions.

----

![5069](../img/cs231n/winter2016/5069.png)

We did the classic.

![5070](../img/cs231n/winter2016/5070.png)

CIFAR-10. 2 FC, 50 neurons.

![5071](../img/cs231n/winter2016/5071.png)

Initialize 2 layer network. Really small network, we can start randomly. We have 10 classes we are using Softmax loss.
$$2.3 = -log(1/10)$$
![5072](../img/cs231n/winter2016/5072.png)

Expect the loss to go up.

![5073](../img/cs231n/winter2016/5073.png)

Good sanity check. Take a small piece of your data, make sure you can over fit. 

![5074](../img/cs231n/winter2016/5074.png)

Loss can go to 0, Accuracy is 100.

![5075](../img/cs231n/winter2016/5075.png)

Do not scale up to your full dataset before you can pass the sanity check.

![5076](../img/cs231n/winter2016/5076.png)
# What is the learning rate that works ? 

![5077](../img/cs231n/winter2016/5077.png)

We can assume the $1e-6$ learning rate was too low.

![5078](../img/cs231n/winter2016/5078.png)

Did you catch that ? Loss barely went down but  accuracy is so good. How ?

![5079](../img/cs231n/winter2016/5079.png)

You start out with fewest scores, when your correct answers become a tiny bit more probable, the accuracy goes up.

![5080](../img/cs231n/winter2016/5080.png)

Huge learning rate, gives errors.

![5081](../img/cs231n/winter2016/5081.png)

What is the ideal learning rate.

![5082](../img/cs231n/winter2016/5082.png)

![5083](../img/cs231n/winter2016/5083.png)

Find the best hyperparameters for the network.

![5084](../img/cs231n/winter2016/5084.png)
## Do coarse -> fine

![5085](../img/cs231n/winter2016/5085.png)
# tip: it is best to optimize in log space! 😍

Because the learning rate is a multiplicative interaction. If we had a linear space, most of the samples would be in a bad region.

![5086](../img/cs231n/winter2016/5086.png)

Do a finer search now.

![5087](../img/cs231n/winter2016/5087.png)

This is worrying, why? If there is a best result in the edge, we might have to adjust the range.

![5088](../img/cs231n/winter2016/5088.png)
## You can grid search instead of random sampling, but this is not good.

Some of the hyperparameters are much more important than others.
# Tip: always use random.

![5089](../img/cs231n/winter2016/5089.png)

This is so much fun.

![5090](../img/cs231n/winter2016/5090.png)

You cannot spray and pray LOL.

![5091](../img/cs231n/winter2016/5091.png)

We might want to consider a bigger learning rate.

![5092](../img/cs231n/winter2016/5092.png)

![5093](../img/cs231n/winter2016/5093.png)

Suddenly starts training.

![5094](../img/cs231n/winter2016/5094.png)
## _lossfunctions.tumblr.com_ LOL 

![5095](../img/cs231n/winter2016/5095.png)

![5096](../img/cs231n/winter2016/5096.png)

We gotta also need to look at accuracy.

Accuracy is absolutely interpretable.

![5097](../img/cs231n/winter2016/5097.png)

Tracking the difference between the scale of your parameters and the scale of your updates to your parameters.

Based on this, you can decide to adjust your learning rate.

![5098](../img/cs231n/winter2016/5098.png)

TLDR for today:

![5099](../img/cs231n/winter2016/5099.png)

We will work on these in the next lecture.

![5100](../img/cs231n/winter2016/5100.png)

Solid performance.
