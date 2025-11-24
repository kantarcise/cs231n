Part of [CS231n Winter 2016](../index.md)

---
## Lecture 6: Training Neural Networks, Part 2

![6001](../img/cs231n/winter2016/6001.png)

By the end of the assignment, you will have a good understanding of all the low-level details of how a ConvNet classifies images.

I am so excited! Here is the [Assignment link again](https://cs231n.github.io/assignments2022/assignment1/).

Training a ConvNet is a four-step process.

- **Loss**: Tells us how well we are classifying at the moment.

- **Backpropagation**: We backpropagate to compute the gradient on all the weights. This gradient tells us how we should nudge every single weight to make better classifications.

- **Update**: We use the gradients to make a small nudge to the weights.

![6002](../img/cs231n/winter2016/6002.png)

There is an entire zoo of activation functions available.

![6003](../img/cs231n/winter2016/6003.png)

### Activation Functions

If you do not use an activation function, your entire network will be a linear sandwich.

Your capacity is equal to that of just a linear classifier.

Activation functions are critical; they provide the non-linearity needed to fit your data.

![6004](../img/cs231n/winter2016/6004.png)

The problem here is: how should we start? Xavier initialization is a reasonable starting point.

![6005](../img/cs231n/winter2016/6005.png)

Batch Normalization (BN) gets rid of many headaches. It reduces the strong dependence on initialization.

![6006](../img/cs231n/winter2016/6006.png)

Here are some tips and tricks for babysitting the learning process.

![6007](../img/cs231n/winter2016/6007.png)

### Today's Agenda

![6008](../img/cs231n/winter2016/6008.png)

![6009](../img/cs231n/winter2016/6009.png)

The process looks like this:

- **Loss**: Tells us how well we are classifying at the moment.
- **Backpropagation**: We backpropagate to compute the gradient on all the weights. This gradient tells us how we should nudge every single weight to make better classifications.
- **Update**: We use the gradients to make a small nudge to the weights.

![6010](../img/cs231n/winter2016/6010.png)

Parameter update is just gradient descent. Can we make it better?

![6011](../img/cs231n/winter2016/6011.png)

### Stochastic Gradient Descent

The classic `.gif` is shown below. In practice, you rarely use vanilla SGD.

![6012](../img/cs231n/winter2016/6012.png)

![parameter update2d](../img/cs231n/winter2016/parameter_update2d.gif)



![parameter update3d](../img/cs231n/winter2016/parameter_update3d.gif)

SGD is the slowest among all of them.

![6013](../img/cs231n/winter2016/6013.png)

There is a big arrow pointing up and a small one pointing right.

![6014](../img/cs231n/winter2016/6014.png)

You are going way too fast in one direction and very slow in the other. This results in jitter.

![6015](../img/cs231n/winter2016/6015.png)

### Momentum
$mu$ is a hyperparameter between 0 and 1.

To solve this problem, we can use momentum.

![6016](../img/cs231n/winter2016/6016.png)

We don't use the learning rate directly; instead, we use velocity to make an update.

Think of a ball rolling around and slowing down over time:

- Gradient is force.
- $mu * v$ is friction.
- $v$ - velocity is initialized with 0.

![6017](../img/cs231n/winter2016/6017.png)

SGD is slower than momentum, as expected. Momentum overshoots the target because it builds up velocity.

![6018](../img/cs231n/winter2016/6018.png)

### Nesterov Momentum
A variation of Momentum Update.

Momentum and gradient step together? We evaluate the gradient at the end of the momentum step.

![6019](../img/cs231n/winter2016/6019.png)

It involves a one-step look-ahead. Evaluate the gradient at the look-ahead step.

![6020](../img/cs231n/winter2016/6020.png)

In theory and in practice, it almost always works better than standard momentum.

![6021](../img/cs231n/winter2016/6021.png)

This is a bit ugly and doesn't fit well in a single API. Normally, we do a forward pass and a backward pass, so we usually have a parameter vector and gradient at that point.

![6022](../img/cs231n/winter2016/6022.png)

You can perform a variable transform.

![6023](../img/cs231n/winter2016/6023.png)

You can check the notes for more details.

![6024](../img/cs231n/winter2016/6024.png)

NAG stands for Nesterov Accelerated Gradient in the graph:

![6025](../img/cs231n/winter2016/6025.png)

NAG curls around much more quickly than SGD with Momentum. 🍓

![parameter update2d](../img/cs231n/winter2016/parameter_update2d.gif)

### Local Minima
As you scale up Neural Networks, the local minima issue goes away; the best and worst local minima get really close.

### AdaGrad
Is it a scale on SGD?

It is very common in practice. Originally developed in convex optimization literature, it was ported to Neural Networks.

```python
cache += dx **2
```

We build a `cache` which is the sum of squared gradients, a giant vector of the same size as the parameter vector.

**Un-centered Second Moment?** This is called a per-parameter adaptive learning rate method.
Every single dimension of the parameter space now has its own learning rate that is scaled dynamically based on the gradients we are seeing.

![6026](../img/cs231n/winter2016/6026.png)

What happens with AdaGrad when updating?

![6027](../img/cs231n/winter2016/6027.png)

We have a large gradient vertically. That large gradient (fast changes) will be added to the cache, and then we end up dividing by larger and larger numbers, so we'll get smaller and smaller updates in the vertical step.

Since we're seeing lots of large gradients vertically, this will decay the learning rate, and we'll make smaller and smaller steps in the vertical direction.

But in the horizontal direction—which is a very shallow direction—we end up with smaller numbers in the denominator. Relative to the Y dimension, we're going to end up making faster progress.

So we have this *equalizing effect of accounting for the steepness*, and in shallow directions, you can actually have a much larger learning rate compared to the vertical directions.

That's AdaGrad.

![6028](../img/cs231n/winter2016/6028.png)

### One problem with AdaGrad: it can decay to a halt.

Your cache ends up building up all the time. You add all these positive numbers to your denominator, so your learning rate just decays towards zero, and you end up stopping learning completely.

That's okay in convex problems, perhaps, where you just have a ball and you decay down to the optimum and you're done.

But in a neural network, things are shuffling around and trying to pick your data. It needs continuous energy to fit your data.

*You don't want it to just decay to a halt.*

`1e-7` is there to prevent the division by zero error. It is also a hyperparameter.

`rmsprop` will forget the gradients from long ago; it is an exponentially weighted sum.

![6029](../img/cs231n/winter2016/6029.png)

### RMSProp
There's a very simple change to AdaGrad that was proposed by Geoff Hinton: `rmsprop`. 🤭

Instead of keeping just the sum of squares in every single dimension, we make that counter *a leaky counter.*

We introduce a decay rate hyperparameter, usually set to something like 0.99. You accumulate the sum of squares, but it leaks slowly with this decay rate.

We still maintain this nice equalizing effect of step sizes in steep or shallow directions, but we won't converge completely to zero updates.

It was just a slide in a Coursera course.

![6030](../img/cs231n/winter2016/6030.png)

People cited this slide. 😅

![6031](../img/cs231n/winter2016/6031.png)

Here is the image again. AdaGrad is blue, RMSProp is black.

![6032](../img/cs231n/winter2016/6032.png)

![parameter update2d](../img/cs231n/winter2016/parameter_update2d.gif)

Usually, in practice when training deep neural networks, `adagrad` stops too early, and `rmsprop` ends up winning out.

### Adam
Combine AdaGrad with Momentum. 🍉

Adam is a recent update that has elements of both.

The Adam optimizer is not necessarily the "best" for all neural networks, but it is a popular and effective choice for many applications. There are several reasons for its popularity:

- **Adaptive learning rate**: Adam optimizer adapts the learning rate for each parameter, which helps in faster convergence and better performance. It combines the advantages of two other popular optimization methods, AdaGrad and RMSProp, by using the first moment estimate (mean) and the second moment estimate (variance) of the gradients.

- **Memory efficiency**: Unlike other adaptive learning rate methods like AdaGrad and RMSProp, Adam only requires the storage of two additional moments (mean and variance) per parameter, making it more memory-efficient.

- **Easy to implement**: Adam is relatively easy to implement, as it only requires the computation of the mean and variance of the gradients, which can be done efficiently using moving averages.

- **Robust performance**: Adam has been shown to perform well on various optimization tasks, including deep neural networks, making it a popular choice among practitioners.

However, it is essential to note that the choice of optimizer depends on the specific problem and the nature of the data. It is always recommended to experiment with different optimizers and tune their hyperparameters to find the best fit for a given task.

![6033](../img/cs231n/winter2016/6033.png)

It's kind of like both together.

In $m$, it sums up the raw gradients, keeping the exponential sum.

In $v$, it keeps track of the second moment of the gradient and its exponential sum.

![6034](../img/cs231n/winter2016/6034.png)

If we compare `rmsprop` with Momentum and Adam:

![6035](../img/cs231n/winter2016/6035.png)

`Beta1` and `Beta2` are hyperparameters. Usually, $beta1 = 0.9$ and $beta2 = 0.995$.

We replace the $dx$ (in the second equation) in RMSProp with $m$, which is the running counter of $dx$.

At any time, you will have noisy gradients. Instead of using those noisy gradients, you use a weighted (decaying) sum of previous gradients, which stabilizes the gradient direction.

The fully complete version is shown below:

![6036](../img/cs231n/winter2016/6036.png)

There is also bias correction, which depends on the time step $t$. Bias correction is only important as Adam is warming up.

![6037](../img/cs231n/winter2016/6037.png)

It depends.

![6038](../img/cs231n/winter2016/6038.png)

You should start with a high learning rate. It optimizes faster. At some point, you will be too stochastic and cannot converge to your minima nicely because you have too much energy in your system and cannot settle down into the nice parts of your loss function.

Decay your learning rate, and you can ride this wagon of decreasing learning rates to do best in all of them.

### Epoch
1 Epoch means you have seen all of the training data once.

### Learning Rate Decays
Step - Exponential - 1/t

These learning rate decays are solid for SGD and Momentum SGD. Adam and AdaGrad are less dependent on them.

Andrej uses Adam for everything now. 🥳

These are all first-order methods because they only use the gradient information of your loss function. When you evaluate the gradient, you know the slope in every single direction.

### Second Order Methods

These provide a better approximation to your loss function. They do not only approximate with the hyperplane (which way we are sloping) but also approximate it with the Hessian, telling you how your surface is curving.

![6039](../img/cs231n/winter2016/6039.png)

- Faster convergence
- Fewer Hyperparameters - No need for a learning rate.

![6040](../img/cs231n/winter2016/6040.png)

- Your Hessian will be gigantic:
- If you have a 100 million parameter network, your Hessian will be `100mil x 100mil`, and you want to invert it.

So, this is not a good idea in Neural Networks.

You can get around inverting the Hessian using BGFS and L-BFGS.

![6041](../img/cs231n/winter2016/6041.png)

These are used in practice.

![6042](../img/cs231n/winter2016/6042.png)

L-BFGS works really well on $f(X)$ functions. In mini-batches, it doesn't work well.

![6043](../img/cs231n/winter2016/6043.png)

Adam is the default. If you have a small dataset, you can look up L-BFGS.

![6044](../img/cs231n/winter2016/6044.png)

What does that mean?

![6045](../img/cs231n/winter2016/6045.png)

Multiple models, average the results. You have to train all of these models, so that is not ideal.

![6046](../img/cs231n/winter2016/6046.png)

You save a checkpoint when you are training.

### Model Ensembles

![6047](../img/cs231n/winter2016/6047.png)

`x_test` is a running sum, exponentially decaying. This `x_test` works better on validation data.

![6048](../img/cs231n/winter2016/6048.png)

### Dropout
A very important technique.

As you are doing a forward pass, you set some neurons randomly to zero.

![6049](../img/cs231n/winter2016/6049.png)

$U1$ is zeros and ones, a binary mask. We apply this mask to hidden layer 1 $H1$ (effectively dropping half of them).

We also do this for the second hidden layer. Do not forget we need to consider this in the backward pass too.

![6050](../img/cs231n/winter2016/6050.png)

#### Motivation

Maybe it will prevent overfitting? All features can have the same strength.

It forces all the neurons to be useful.

![6051](../img/cs231n/winter2016/6051.png)

#### Feature Co-adaptation

![6052](../img/cs231n/winter2016/6052.png)

You cannot rely on a single feature.

![6053](../img/cs231n/winter2016/6053.png)

A dropped-out neuron will not have connections to the previous layer, as if it were not there.

You are sub-sampling a part of your Neural Network, and you are only training that neural network on that single example that you have at that point in time.

You want to apply stronger dropout where there is a huge number of parameters.

In practice, you do not use dropout at the start of Convolutional Neural Networks; you scale the dropout over time.

#### Instead of dropping gradients, you can drop weights. That is called DropConnect.

![6054](../img/cs231n/winter2016/6054.png)

We would like to integrate out all of the noise. You can try all binary masks and average the result, but that is not really efficient.

![6055](../img/cs231n/winter2016/6055.png)

You can approximate this with Monte Carlo.

In an ideal world, you do not want to leave any neurons behind.

![6056](../img/cs231n/winter2016/6056.png)

Can we use expectation?

![6057](../img/cs231n/winter2016/6057.png)

During testing, a linear neuron will give, in expectation, **half** of what it gives at training time.

That half comes from the half of the units we dropped.

![6058](../img/cs231n/winter2016/6058.png)

If we do not do this, we will end up having too large of an output compared to what we had in expectation at training time. Things will break in the NN, as they are not used to seeing such large outputs from the neurons.

#### Test Time Scaling

![6059](../img/cs231n/winter2016/6059.png)

In this example, $p$ can be $0.5$.

![6060](../img/cs231n/winter2016/6060.png)

Do not forget to also backpropagate the masks.

#### Inverted Dropout

![6061](../img/cs231n/winter2016/6061.png)

We select $p$ each time we have a mini-batch.

Even though there is randomness in the exact amount of dropout, we still use 0.5.

![6062](../img/cs231n/winter2016/6062.png)

Implement what you learn. Fast. Deep Learning Summer School [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/).

![6063](../img/cs231n/winter2016/6063.png)

Go through the notes. Here is the [link for it](https://cs231n.github.io/neural-networks-3/).

![6064](../img/cs231n/winter2016/6064.png)

### Convolutional Neural Networks

LeNet-5 - 1980.

![6065](../img/cs231n/winter2016/6065.png)

Fei Fei Li told us about this. Here is a video on [the experiment](https://www.youtube.com/watch?v=OGxVfKJqX5E).

This is one neuron in the `V1` cortex. In a particular orientation, neurons get excited about edges.

![6066](../img/cs231n/winter2016/6066.png)

Nearby cells in the visual cortex process nearby areas in your visual field. Locality is preserved in processing.

![6067](../img/cs231n/winter2016/6067.png)

The visual cortex has a hierarchical organization, going from simple cells to complex cells through layers.

![6068](../img/cs231n/winter2016/6068.png)

A layered architecture with these local receptive cells looks at a part of the input.

![6069](../img/cs231n/winter2016/6069.png)

There was no backpropagation.

Yann LeCun built on top of this knowledge. He kept the rough architecture layout and trained the network using backpropagation.

![6070](../img/cs231n/winter2016/6070.png)

AlexNet. In 2012, it won the ImageNet Challenge.

![6071](../img/cs231n/winter2016/6071.png)

ConvNets can classify images.

They are really good at retrieval, showing similar images.

![6072](../img/cs231n/winter2016/6072.png)

They can do detection.

![6073](../img/cs231n/winter2016/6073.png)

They are used in cars. You can do perception of things around you.

![6074](../img/cs231n/winter2016/6074.png)

ConvNets are really good face detectors, like for tagging friends in Facebook.

Google is really interested in detecting street numbers.

![6075](../img/cs231n/winter2016/6075.png)

They can detect poses and play computer games.

![6076](../img/cs231n/winter2016/6076.png)

They can work on cells. They can read Chinese. They can recognize street signs.

![6077](../img/cs231n/winter2016/6077.png)

They can recognize speech (a non-visual application). They can be used with text too.

![6078](../img/cs231n/winter2016/6078.png)

Specific types of whales. Satellite image analysis.

![6079](../img/cs231n/winter2016/6079.png)

They can do image captioning.

![6080](../img/cs231n/winter2016/6080.png)

They can do DeepDream. ImageNet has a lot of dogs, so they hallucinate dogs.

![6081](../img/cs231n/winter2016/6081.png)

I will not explain this one.

![6082](../img/cs231n/winter2016/6082.png)

From an image, you can get results with a ConvNet almost equal to a monkey's IT Cortex.

![6083](../img/cs231n/winter2016/6083.png)

We show a lot of images to both the monkey and the ConvNet.

If you look at how images are represented in the brain and the ConvNet, the mapping is really, really similar.

![6084](../img/cs231n/winter2016/6084.png)

#### How do they work?

![6085](../img/cs231n/winter2016/6085.png)

Next class.
