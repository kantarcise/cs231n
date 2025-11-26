Part of [CS231n Winter 2016](../index.md)

---
## Lecture 7: Convolutional Neural Networks

Two weeks to go on Assignment 2.

![7001](../img/cs231n/winter2016/7001.png)

Project Proposal? [About Project](http://cs231n.stanford.edu/project.html)

The four-step process is still relevant.

![7002](../img/cs231n/winter2016/7002.png)

How did we update parameters?

![7003](../img/cs231n/winter2016/7003.png)

Dropout was casually introduced by Geoffrey Hinton.

![7004](../img/cs231n/winter2016/7004.png)

LeNet is a classic architecture.

![7005](../img/cs231n/winter2016/7005.png)

Hubel and Wiesel's experiments.

![7006](../img/cs231n/winter2016/7006.png)

Feature hierarchy.

![7007](../img/cs231n/winter2016/7007.png)

![7008](../img/cs231n/winter2016/7008.png)

We start with a $32x32x3$ CIFAR-10 image.

It has 3 channels, so the volume of activations is 3 deep. This corresponds to the 3rd dimension of the volume.

![7009](../img/cs231n/winter2016/7009.png)

**Convolutional Layer**: A core building block.

A filter with a depth of 3 will cover the full depth of the input volume. However, it is spatially small ($5x5$).

![7010](../img/cs231n/winter2016/7010.png)

We always extend through the full depth of the input volume.

![7011](../img/cs231n/winter2016/7011.png)

We will learn $w$. We are going to slide the filter over the input volume.

As we slide, we perform a 75-dimensional dot product.

![7012](../img/cs231n/winter2016/7012.png)

This sliding process results in an **activation map**.

#### Activation Map Size

Because we slide the filter from index 0 to 4 on the input image, we can place the filter in 28x28 distinct locations.

![7013](../img/cs231n/winter2016/7013.png)

We will actually have a filter bank. Different filters will result in different activation maps.

![7014](../img/cs231n/winter2016/7014.png)

6 filters will result in 6 activation maps.

![7015](../img/cs231n/winter2016/7015.png)

#### Output Dimensions

After all the convolutions, we will have a new image sized $28x28x6$!

![7016](../img/cs231n/winter2016/7016.png)

We will have these convolutional layers, which will have a certain number of filters. These filters will have a specific spatial extent (e.g., 5x5). This conv layer will slide over the input and produce a new image. This will be followed by a ReLU and another conv layer.

![7017](../img/cs231n/winter2016/7017.png)

The filters now have to be $5x5x6$.

#### Input Depth Matching

These filters are initialized randomly. They will become the parameters in our ConvNet.

![7018](../img/cs231n/winter2016/7018.png)

When you look at the trained layers, the first layers represent low-level features: color pieces, edges, and blobs.

The first layers will look for these features in the input image as we convolve through it.

As you go deeper, we perform convolution on top of convolution, doing dot products over the outputs of the previous conv layer.

It will put together all the color/edge pieces, making larger and larger features that the neurons will respond to.

For example, mid-level layers might look for circles.

And in the end, we will build object templates and high-level features.

In the leftmost picture, these are raw weights ($5x5x3$ array).

In the middle and right, these are visualizations of what those layers are responding to in the original image.

![7019](../img/cs231n/winter2016/7019.png)

This is pretty similar to what Hubel and Wiesel imagined: a bar of a specific orientation leads to more complex features.

![7020](../img/cs231n/winter2016/7020.png)

A small piece of a car as input.

32 filters of size 5x5 in the first convolutional layer.

Below are example activation maps. **White** corresponds to high activation, and **black** corresponds to low activation (low numbers).

Where the blue arrow points to orange stuff in the image, the activation shows that the filter is **happy** about that part.

![7021](../img/cs231n/winter2016/7021.png)

A layout like this:

### Architecture Overview

Also, a Fully Connected layer at the end.

Every row is an activation map. Every column is an operation.

#### ReLU Layer

The image feeds into the left side. We do convolution, thresholding (ReLU), then another Conv, another ReLU, then pooling...

Piece by piece, we create these 3D volumes of higher and higher abstraction. We end up with a volume connected to a large FC layer.

The last matrix multiplication will give us the class scores.

![7022](../img/cs231n/winter2016/7022.png)

#### Filter Count

We are only concerned about spatial dimensions at this point.

![7023](../img/cs231n/winter2016/7023.png)

One at a time.

![7024](../img/cs231n/winter2016/7024.png)

One at a time.

![7025](../img/cs231n/winter2016/7025.png)

One at a time.

![7026](../img/cs231n/winter2016/7026.png)

One at a time.

![7027](../img/cs231n/winter2016/7027.png)

We can use a stride of 2, which is a hyperparameter.

![7028](../img/cs231n/winter2016/7028.png)

We move two steps at a time.

![7029](../img/cs231n/winter2016/7029.png)

We are done in fewer steps!

![7030](../img/cs231n/winter2016/7030.png)

Can we use a stride of 3?

![7031](../img/cs231n/winter2016/7031.png)

No, we cannot.

![7032](../img/cs231n/winter2016/7032.png)

This simple formula gives you possible selections. The result should always be an integer.

![7033](../img/cs231n/winter2016/7033.png)

We can pad! Padding is also a hyperparameter.

![7034](../img/cs231n/winter2016/7034.png)

If we pad with 1, we can get an output of the ***same size***.

#### Spatial Preservation

![7035](../img/cs231n/winter2016/7035.png)

#### Padding Strategy

![7036](../img/cs231n/winter2016/7036.png)

If we do not pad, the size will shrink! We do not want that, as we will have many layers.

![7037](../img/cs231n/winter2016/7037.png)

10 filters with $5x5x3$ shape.

![7038](../img/cs231n/winter2016/7038.png)

The padding is correct, so the spatial size will not change.

10 filters will generate 10 different activation maps.

The output is shaped $32x32x10$.

![7039](../img/cs231n/winter2016/7039.png)

#### Parameter Counting

![7040](../img/cs231n/winter2016/7040.png)

Each filter has $5*5*3$ parameters plus a single bias. So the total is $10 * 76 = 760$.

![7041](../img/cs231n/winter2016/7041.png)

Here is the summary so far:

![7042](../img/cs231n/winter2016/7042.png)

#### Filter Hyperparameters
- Number of filters
- The spatial extent of the filters - $F$
- The stride - $S$
- The amount of zero padding - $P$

![7043](../img/cs231n/winter2016/7043.png)

We can compute the size of the activation output with the formula. The depth will be the number of filters $K$. $F$ is usually odd.

The total number of parameters will depend on input depth, filter size, and bias.

$K$ is usually chosen as a power of 2 for computational reasons. Some libraries use special subroutines when they see powers of 2.

![7044](../img/cs231n/winter2016/7044.png)

We can use $1x1$ convolutions. You are still doing a lot of computation; you are just not merging information spatially.

#### Zero Padding Rationale

#### Non-Square Inputs

We will see how to work with non-rectangular images later.

![7045](../img/cs231n/winter2016/7045.png)

#### Terminology

The API of `SpatialConvolution` in Torch:

- `nInputPlane`: The depth of the input layer.

- `nOutputPlane`: How many filters you have.

- `kW`, `kH`: Kernel width and height.

- `dW`, `dH`: Step size (stride).

- `padW`, `padH`: The padding you want.

> This is referring to Lua Torch (Torch7), which was the predecessor to the modern PyTorch, with this [Conv2d class here](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

![7046](../img/cs231n/winter2016/7046.png)

It is the same in `Caffe`.

![7047](../img/cs231n/winter2016/7047.png)

It is the same in `Lasagne`.

### Biological Perspective

![7048](../img/cs231n/winter2016/7048.png)

With this filter, we end up with one number in a convolution.

![7049](../img/cs231n/winter2016/7049.png)

The output of the filter at this position is just a neuron fixed in space, looking at a small part of the input, computing $w^T x + b$.

It has no connections to other parts of the image, hence local connectivity.

![7050](../img/cs231n/winter2016/7050.png)

We sometimes refer to the neuron's receptive field as the size of the filter (the region of the input the filter is looking at).

In a single activation map (28x28 grid), these neurons share parameters (because one filter computes all the outputs), so all the neurons have the same weights $w$.

#### Weight Sharing

We have several filters, so **spatially they share weights**, but across **depth**, these are all different neurons.

![7051](../img/cs231n/winter2016/7051.png)

A nice advantage of both local connectivity and spatial parameter sharing is that it basically controls the capacity of the model.

It makes sense that neurons would want to compute similar things. For example, if they are looking for edges, a vertical edge in the middle of an image is just as useful anywhere else spatially.

It makes sense to share those parameters spatially as a way of controlling overfitting.

![7052](../img/cs231n/winter2016/7052.png)

We have covered Conv and ReLU layers.

### Pooling Layer

![7053](../img/cs231n/winter2016/7053.png)

The Conv layer usually preserves the spatial size (with padding).

The spatial shrinking is done by pooling.

#### Motivation

The most common method is max pooling.

![7054](../img/cs231n/winter2016/7054.png)

It reduces the size by half on all activation maps. Average pooling does not work as well.

![7055](../img/cs231n/winter2016/7055.png)

We need to know the filter size and stride. $2x2$ with stride $2$ is common.

![7056](../img/cs231n/winter2016/7056.png)

The depth of the volume does not change.

### Fully Connected Layer

![7057](../img/cs231n/winter2016/7057.png)

With 3 pooling layers ($2x2$, stride 2), we go from 32 -> 16 -> 8 -> 4.

At the end, we have a $4x4x10$ volume of activations after the last pooling.

That goes into the Fully Connected layer.

### Demo

Website [here](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html). It achieves 80% accuracy for CIFAR-10 in JavaScript!

It uses 6 or 7 nested loops. The V8 engine in Chrome is good, so JS is fast.

![7058](../img/cs231n/winter2016/7058.png)

All running in the browser.

---

#### Stacking Layers

Why is that we are stacking layers? 🤔

Because we want to do dot products, and we can backpropagate through them efficiently.

#### Batch Dimensions

If you are working with image batches, all the volume between Convnet's are 4D arrays. If single image, 3D arrays.

#### Visualization

Intermediate filters are not properly visualized. Yann LeCun did what the neurons are responding to. 


#### Pooling Rationale

When you do pooling, you throw away some spatial information because you want to eventually get the scores out.

#### Boundary Effects

Because of padding, the statistics of border is different than center, we do not worry about it.

### Backpropagation Compatibility

Anything you can back propagate through, you can put in a ConvNet / Neural Net.

### Case Studies

#### LeNet-5

Figure from the paper. 6 filters, all $5x5$, with sub-sampling (max pooling).

![7059](../img/cs231n/winter2016/7059.png)

#### AlexNet

It won the ImageNet challenge.  60 Million Parameters.

The input is large.

Two separate streams? Alex had to split the convolutions onto 2 separate GPUs.

Let's imagine if it had a single stream.

![7060](../img/cs231n/winter2016/7060.png)

The output volume will be: $55x55x96$, because we have 96 kernels/filters.

![7061](../img/cs231n/winter2016/7061.png)

Total parameters: every filter is $11x11x3$ x 96 roughly.

We are not even sure what Alex did. 😅

The input image is $224x224$, but for the math to add up, the input should be $227x227$.

![7062](../img/cs231n/winter2016/7062.png)

After pooling? Half of the spatial size, so $27x27x96$.

![7063](../img/cs231n/winter2016/7063.png)

How many parameters are in the pooling layer?

![7064](../img/cs231n/winter2016/7064.png)

0 - only Conv layers have parameters.

![7065](../img/cs231n/winter2016/7065.png)

Summary:

![7066](../img/cs231n/winter2016/7066.png)

Full architecture:

![7067](../img/cs231n/winter2016/7067.png)

This is a classic sandwich. Sometimes filter sizes change. We backpropagate through all of this.

![7068](../img/cs231n/winter2016/7068.png)

First use of ReLU, used normalization layers (not used anymore), used dropout only on the last fully connected layers, and an ensemble of 7 models.

#### ZFNet

Built on top of AlexNet.

$11x11$ stride 4 was too drastic, so they changed to $7x7$ filters.

They used more filters in Conv 3, 4, and 5.

The error became 14.8%. The author of this paper founded a company called Clarifai and reported 11% error.

Here is the [company about.](https://www.clarifai.com/company/about)

> Founded in 2013 by Matthew Zeiler, Ph.D., a foremost expert in machine learning, Clarifai has been a market leader since winning the top five places in image classification at the ImageNet 2013 competition.

**Top-5 Error**

There are 1000 classes, and we give the classifier 5 chances to guess. 😌

![7069](../img/cs231n/winter2016/7069.png)

#### VGGNet

140 Million Parameters. They have different types of architectures. They decided to use a single set of filters. The question is:

**Layer Count**

Turns out, 16 layers performed the best. They dropped the error to 7.3%.

![7070](../img/cs231n/winter2016/7070.png)

This is the full architecture:

**Spatial Reduction**

Spatially the volumes get smaller, number of filters are increasing

![7071](../img/cs231n/winter2016/7071.png)

**Memory Usage**

![7072](../img/cs231n/winter2016/7072.png)

If we add up all the numbers, it's 24M. If we use float32, that's 93 MB of memory for intermediate activation volumes per image.

That is maintained in memory because we need it for backpropagation.

Just to represent 1 image, it takes 93 MB of RAM ONLY for the FORWARD pass. For the backward pass, we also need the gradients, so we end up with a 200 MB footprint.

**Parameter Count**

Most memory is in early Conv layers; most parameters are in late FC layers.

We found that these huge Fully Connected layers are not necessary.

**Average Pooling**

Instead of FC on $7x7x512$, you can average on $7x7$ and make it a single $1x1x512$, which works just as well.

#### GoogleNet

The key innovation here was the **Inception** module. Instead of using direct convolutions, they used inception modules.

A sequence of inception modules makes up GoogleNet. You can read the paper.

It won the 2014 challenge with 6.7% error.

![7073](../img/cs231n/winter2016/7073.png)

At the very end, they had $7x7x1024$ and they did an average pool! That means much fewer parameters!

![7074](../img/cs231n/winter2016/7074.png)

#### ResNet

![7075](../img/cs231n/winter2016/7075.png)

Here is what the history looks like.

![7076](../img/cs231n/winter2016/7076.png)

More layers. You have to be careful how you increase the number of layers.

**Plain vs ResNets**

A 56-layer network performs worse than a 44-layer network. Why?

In ResNets, increasing the number of layers will always result in better results.

![7077](../img/cs231n/winter2016/7077.png)

**Runtime Performance**

At Runtime, it is actually faster than a VGGnet  - how ?

![7078](../img/cs231n/winter2016/7078.png)

This is a plain net below:

We will have skip connections.

You take a $224x224$ image, pool by a huge factor, and work spatially on $56x56$. It's still really good.

Depth comes at the cost of spatial resolution very early on, because depth is to their advantage.

![7079](../img/cs231n/winter2016/7079.png)

In a plain net, you have some function $f(x)$ you are trying to compute. You transform your representation, have a weight layer, threshold it, and so on.

In a ResNet, your input flows in. But instead of computing how to transform your input into $f(x)$, you compute what to add to your input to transform it into $F(x)$.

You compute a delta on top of your original representation instead of a new representation right away, which would discard the original information about $X$.

**Delta Modulation Analogy**

> In analogy, you can think of delta modulation as encoding the difference between successive samples (input and output), somewhat akin to how the ResNet architecture focuses on learning the residual (difference) between input and output to improve learning efficiency. Both methods leverage this residual information for better representation or reconstruction.

You are computing just these deltas to these $x$'s.

If you think about the gradient flow in a ResNet, when a gradient comes, it performs addition (remember, addition distributes the gradient to all of its children). The gradient will flow to the top, skipping over the straight part.

![7080](../img/cs231n/winter2016/7080.png)

You can train right away, really close to the image, to the first Conv Layer.

![7081](../img/cs231n/winter2016/7081.png)

These are the commonly used hyperparameters.

- Batch norm layers will allow you to get away with a bigger learning rate.

![7082](../img/cs231n/winter2016/7082.png)

- Using $1x1$ Convs in clever ways.

![7083](../img/cs231n/winter2016/7083.png)

This is the whole architecture; Andrej skipped it in the interest of time.

![7084](../img/cs231n/winter2016/7084.png)

This was on the cover of AlphaGo.

![7085](../img/cs231n/winter2016/7085.png)

This was a convolutional network!

![7086](../img/cs231n/winter2016/7086.png)

The input is $19x19x48$ because they are using 48 different features based on the specific rules of Go. You can understand what is going on when you read the paper.

Other Go Deep Learning player: [CrazyStone](https://www.remi-coulom.fr/CrazyStone/)

![7087](../img/cs231n/winter2016/7087.png)

- The trend is to get rid of Pooling and Fully Connected Layers.

- Smaller filters and deeper architectures.

Done with lecture 7!
