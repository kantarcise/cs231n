Part of [CS231n Winter 2016](../index.md)

---
## Lecture 9: Understanding ConvNets

This is one of Andrej's favorite lectures to give.

![9001](../img/cs231n/winter2016/9001.png)

Assignment 2 is almost due. The midterm is next week. They just released the winning weights.

![9002](../img/cs231n/winter2016/9002.png)

There is a wide variety of application domains for CNNs.

![9003](../img/cs231n/winter2016/9003.png)

We saw how ConvNets work. We covered all the basics.

![9004](../img/cs231n/winter2016/9004.png)

We looked at a lot of different Computer Vision tasks, including R-CNN, Fast R-CNN, Faster R-CNN, and YOLO.

Multiple heads are placed on top of a ConvNet; some heads do classification, and some do regression. They are all trying to solve the problem at hand.

![9005](../img/cs231n/winter2016/9005.png)

### Understanding ConvNets

We will go over all of these bullet points.

Perhaps the simplest way to understand what a ConvNet is doing is to look at its raw activations.

In a CNN, we pass an image into the bottom, and we get activation volumes in between.

![9006](../img/cs231n/winter2016/9006.png)

We can select a random neuron—say, on the pool 5 layer—pipe a lot of images into the ConvNet, and see what excites that neuron the most.

Some of them like dogs, some like flags. Some like text, and some like lights.

### Visualizing Weights

On the first layer, we can visualize the weights. In the first layer of convolution, we have a filter bank that we slide over the image, so we can visualize the raw filter weights.

When weights are not directly connected to the image, visualization doesn't really make sense. It only makes sense in the first layer.

![9007](../img/cs231n/winter2016/9007.png)

You can still do it, but it doesn't make as much sense.

![9008](../img/cs231n/winter2016/9008.png)

Pretty much anything you throw at an image to learn a feature will result in these Gabor-like features (Gabor filters, a mathematical function used in signal processing), regardless of the algorithm.

The inverse is actually hard to do for the first layer. The example Andrej gave is PCA; it doesn't give Gabor-like features, but rather sinusoids.

![9009](../img/cs231n/winter2016/9009.png)

### Global Representation

We looked at filters and weights.

Another way to look at it is to pass a lot of images through the ConvNet and look at the **FC-7 Features**.

These are 4096 numbers just before the classifier. These numbers summarize the content of the image. These are codes we can use.

![9010](../img/cs231n/winter2016/9010.png)

### t-SNE Visualization

You give it a collection of high-dimensional vectors, and it finds an embedding in 2D such that points that are nearby in the original space are nearby in the embedding.

It does this in a clever way that gives us really nice-looking pictures.

Below you can see the embeddings for MNIST:

![9011](../img/cs231n/winter2016/9011.png)

#### Embedding Proximity 🍉

![9012](../img/cs231n/winter2016/9012.png)

Here is [the link](https://cs.stanford.edu/people/karpathy/cnnembed/), and the full image is below. All the boats are close, all the spaghetti is close, as are all the dogs and animals.

![cnn embed full 4k](../img/cs231n/winter2016/cnn_embed_full_4k.jpg)

This is what ConvNets consider similar.

### Occlusion Experiments 🐔

*Visualizing and Understanding Convolutional Networks* by Matthew D. Zeiler and Rob Fergus, published in 2013.

The main idea behind occlusion experiments is to understand which parts of an input image are crucial for a CNN's decision-making process. This is achieved by systematically occluding (covering) different parts of the input image and observing how the network's output changes as a result.

- A patch of zeros (occluder) is shown in grey.

- We slide it over the image.

- As we do that, we look at the probability of the class and how it varies as a function of the spatial location of that occluder.

![9013](../img/cs231n/winter2016/9013.png)

We would expect the probability to go down when we cover up the dog. That is basically what happens.

We get a kind of heat map from this.

The same applies to the dog and the car wheel.

In the last picture, interestingly, when you cover the person on the left, the probability goes up!

This is because the ConvNet is not sure if the class is there or not. When you remove the person, the ConvNet becomes more sure.

![9014](../img/cs231n/winter2016/9014.png)

### DeepVis Toolbox

Jason Yosinski! Running a ConvNet in real-time, you can summon your camera feed and play with the ConvNet to see all the activations.

2 Methods:

- Deconvolution
- Optimization Based

Watching [the video](https://www.youtube.com/watch?v=AgkfIQ4IGaM)!

Neural Networks are really good at classification, thanks to Convolutional Neural Networks.

**Conv Layer 1**: Light to dark or dark to light; different layers like different things. Some layers like heads and shoulders and ignore the rest of the body.

Some layers activate when they see cats. Some activate on non-smooth (wrinkled) clothes (not the clothing itself). Some just like text.

![9015](../img/cs231n/winter2016/9015.png)

You can investigate and debug ConvNets in real-time.

#### Deconv Approach

How would you compute the gradient of any neuron with respect to the image?

![9016](../img/cs231n/winter2016/9016.png)

Normally, we have a computational graph, pass the image through, and get a loss at the end. We start with $1.00$ in our computational graph because the gradient of loss with respect to loss is 1.

We want to backpropagate and find the influence of all the inputs on that output.

**Gradient Computation 🤨**

![9017](../img/cs231n/winter2016/9017.png)

- We forward pass until some layer.
- We have activations for that layer.
- We are interested in some specific neuron.
- We zero out all the gradients in that layer except for the neuron's gradient we are interested in; we set that neuron's gradient to $1.00$.
- Run backward from that point on (backpropagate).
- When you backpropagate to the image, you will find the gradient of the image with respect to any arbitrary neuron by playing with the gradients.

![9018](../img/cs231n/winter2016/9018.png)

You will find something like this.

![9019](../img/cs231n/winter2016/9019.png)

The Deconv approach changes the backward pass a bit; it's not entirely clear why.

**Guided Backpropagation**

Much cleaner images, showing the cat's face.

This is a figure from the paper: The image goes through layers, and we get an activation map at some place.

We zero out all the gradients except the one we are interested in.

![9020](../img/cs231n/winter2016/9020.png)

To get your Deconv to give you nice images, we will run backpropagation, but we will change the backprop in the **ReLU** layer!

You can see in **c)** that we have the activation, just like we described.

If your input was negative, you block the gradient in the backward pass, as per ReLU.

![9021](../img/cs231n/winter2016/9021.png)

In guided backpropagation, we change the backward ReLU in the following way:

We compute what we had before, but we add a term that says we only backpropagate through our ReLU neurons where the ReLU neurons have a positive gradient.

Normally, we would pass any gradients corresponding to the ReLU neuron that had less than zero input. Now, in addition to that, we block out all the gradients corresponding to negative gradients.

**Interpretation**

- We are trying to compute the influence of the input on some arbitrary neuron in the ConvNet.
- A negative gradient means that the ReLU neuron has a negative influence on the neuron we are investigating.
- By doing that, we only pass through gradients that have an entirely positive influence on the activations.

**Backpropagation Dynamics**

![9022](../img/cs231n/winter2016/9022.png)

The reason we get weird images (like the one with the cat) is that some influences are positive and some are negative from every single pixel to the neuron we are investigating.

In **guided backpropagation**, we only use positive influences—only the positive gradients from the ReLU.

You get much cleaner images.

![9023](../img/cs231n/winter2016/9023.png)

Another approach is `DeconvNet`:

- It ignores the ReLU gradient.
- It just passes through the positive gradient; it does not care if the activations coming to the ReLU are positive or negative.
- It works well.

![9024](../img/cs231n/winter2016/9024.png)

This is a similar idea to guided backpropagation.

![9025](../img/cs231n/winter2016/9025.png)

From Layer 3 onwards, you see shapes.

![9026](../img/cs231n/winter2016/9026.png)

In the third row, third column, you see a human face as red. This means the gradient is telling you that if you made this person's face redder, it would have a locally positive effect on this neuron's activation.

Layer 4 starts to form objects.

![9027](../img/cs231n/winter2016/9027.png)

Andrej is not a big fan of the DeConv approach. You get pretty images, but that's about it.

#### Optimization to Image

We will do a bit more work compared to the DeConv route.

We are going to try to **optimize the image** while keeping the Convolutional Neural Network fixed.

We are going to try to maximize an arbitrary score in the ConvNet.

![9028](../img/cs231n/winter2016/9028.png)

We are trying to find an image $I$ such that your score is maximized, subject to some regularization on $I$.

- L2 Regularization: Discourages parts of your input from being too large.

![9029](../img/cs231n/winter2016/9029.png)

We start with a Zero Image. We feed it into a ConvNet.

We set the gradient at that point to be all 0s, except for a 1 at the neuron we are interested in.

This is just normal backpropagation.

![9030](../img/cs231n/winter2016/9030.png)

We do a forward pass, a backward pass, and then updates.

![9031](../img/cs231n/winter2016/9031.png)

Iterate this over and over to optimize the image.

![9032](../img/cs231n/winter2016/9032.png)

**Geese Example 🥰**

![9033](../img/cs231n/winter2016/9033.png)

Another way of interpreting the gradient signal at the image is from the following paper:

**Area of Influence**

![9034](../img/cs231n/winter2016/9034.png)

They forward the image (the dog), set the gradient to 1, and do backpropagation.

You arrive at your image gradient, and they squish it through channels with a $max$ function.

What would you expect?

![9035](../img/cs231n/winter2016/9035.png)

In the black parts of the image, if you wiggle a black pixel, the score for that image does not change at all. The ConvNet does not care about it.

So the gradient signal can be used (in a **GrabCut** segmentation) as a measure of the area of influence on the input image.

![9036](../img/cs231n/winter2016/9036.png)

You can crop images just based on the gradient signal.

Seems suspicious -> Cherry-picked examples...

![9037](../img/cs231n/winter2016/9037.png)

We were maximizing the full score and optimizing the image. We can do this for any arbitrary neuron.

![9038](../img/cs231n/winter2016/9038.png)

We have been using L2 penalty so far. Is there a better way?

- Ignore the penalty.
- Do forward and backward passes.
- Blur the image a bit (this prevents the image from accumulating high frequencies).
- This blurring will help you get cleaner visualizations for classes.

![9039](../img/cs231n/winter2016/9039.png)

This looks a bit better. 4 different results with 4 different initializations.

![9040](../img/cs231n/winter2016/9040.png)

You can go down layers to see.

![9041](../img/cs231n/winter2016/9041.png)

In Layer 5, there is some part of an ocean.

![9042](../img/cs231n/winter2016/9042.png)

These just come out of the optimization. This is what these neurons really like to see.

#### Effective Receptive Field

In the first layer of VGG, it is just $3x3$. As you go down, the effective receptive field gets bigger. So you see neurons that are functions of the entire image.

#### Information Content

Can you invert the image with just **the code**?

![9043](../img/cs231n/winter2016/9043.png)

- We are given a particular feature.
- We want to find an image that best matches that code.

Instead of maximizing any arbitrary feature, we just want to have a specific feature and exactly match it in every single dimension.

![9044](../img/cs231n/winter2016/9044.png)

When you run the optimization, you will get something like this.

![9045](../img/cs231n/winter2016/9045.png)

You can do reconstruction at any place in the ConvNet. The example below is even better than our first one.

The bird location is pretty accurate, so this is proof that the code is rich in information.

![9046](../img/cs231n/winter2016/9046.png)

You can also look at a single image and see how much information is thrown away as you move forward.

You can compare reconstruction at different layers. When you are very close to the image, you can do a very good job of reconstruction.

![9047](../img/cs231n/winter2016/9047.png)

A flamingo example:

![9048](../img/cs231n/winter2016/9048.png)

You can get really funky images as you try to optimize the image. It's 100 lines of code in a Python notebook.

![9049](../img/cs231n/winter2016/9049.png)

This is based on an Inception network. We choose the layer we want to dream at.

`make_step` will be called repeatedly.

![9050](../img/cs231n/winter2016/9050.png)

We forward pass the network, call the objective on the layer we want to dream at, and then do a backward pass.

![9051](../img/cs231n/winter2016/9051.png)

You have a ConvNet. You pass the image through to some layer where you want to dream. The gradients at that point become ***exactly identical*** to the activations at that point. Then you backpropagate to the image.

There are so many features that really care about dogs because there are so many of them in the training data for ImageNet. A large portion of ConvNet features really like dogs.

We want to boost what we know. If a cloud resembles a dog, the image will be refined to be more dog-like.

![9052](../img/cs231n/winter2016/9052.png)

Funky things.

![9053](../img/cs231n/winter2016/9053.png)

If you DeepDream lower, the features are more like edges and shapes.

![9054](../img/cs231n/winter2016/9054.png)

Funny videos.

![9055](../img/cs231n/winter2016/9055.png)

### Neural Style

You can take a picture and render it in a different style.

This is achieved by **Optimization on the raw image with ConvNets**.

![9056](../img/cs231n/winter2016/9056.png)

Examples:

![9057](../img/cs231n/winter2016/9057.png)

We have a content image and a style image.

We pass the content image into the ConvNet. We hold the activations as they represent the content.

![9058](../img/cs231n/winter2016/9058.png)

We take the style image and pass it through the ConvNet.

Instead of keeping track of the raw activations, the paper authors found that the style was not in the raw activations but in their pairwise statistics.

We got a $224x224x64$ activation at the Conv1 Layer. We want some fibers from it. $64x64$ (Gram matrices) is what we want.

**Feature Correlations**

![9059](../img/cs231n/winter2016/9059.png)

We will do this on every Conv layer.

- We want to match the content (all the actual activations from content) and style (the Gram matrices).
- These 2 objectives are fighting it out.
- In practice, we run content in Layer 5 (a single layer) and use many more layers for style.

![9060](../img/cs231n/winter2016/9060.png)

This is best optimized with L-BFGS. We do not have a huge dataset, everything fits in memory, so second-order methods (instead of Adam, AdaGrad) work really well here.

---

### Adversarial Examples

We saw all the optimizations on the image.

![9061](../img/cs231n/winter2016/9061.png)

You can make a school bus, or anything, into an ostrich.

We get the gradient on that image for the Ostrich class.

We forward the image. We set all gradients to 0 except for the class we want (ostrich). We do a backward pass, and we get a gradient of what to change in the image to make it more like an Ostrich.

![9062](../img/cs231n/winter2016/9062.png)

The distortion you need is really small. You can turn anything into anything.

You can start from random noise.

![9063](../img/cs231n/winter2016/9063.png)

You can use weird geometric shapes.

![9064](../img/cs231n/winter2016/9064.png)

This is not really new; this happened before.

HOG representation is identical, but the images are so different from each other.

![9065](../img/cs231n/winter2016/9065.png)

#### Manifold Hypothesis

Images are super high-dimensional objects (150,000-dimensional space).

Real images that we train on have a special statistical structure and are constrained to tiny manifolds in that space.

We train ConvNets on these. These ConvNets work really well on that tiny manifold, where the statistics of images are actually image-like.

We are putting these linear functions on top of it. We only know a little part; there are a lot of shadows.

![9066](../img/cs231n/winter2016/9066.png)

Let's just work with logistic regression.

$x$ is 10-dimensional. $w$ is a multi-dimensional vector and $b$ is bias.

We put that through a sigmoid. We interpret the output of that sigmoid as the probability that the input $x$ is of class 1.

We compute the score with this classifier. The input is class $1$ if the score is greater than 0 (or equivalently, if the sigmoid output is greater than $0.5$).

![9067](../img/cs231n/winter2016/9067.png)

No bias example:

![9068](../img/cs231n/winter2016/9068.png)

Just the dot product of the vectors. With this setting of weights, this classifier thinks it's 95% class 0.

![9069](../img/cs231n/winter2016/9069.png)

#### Adversarial Perturbations

We want to slightly modify $x$ and **confuse the classifier**.

![9070](../img/cs231n/winter2016/9070.png)

We want to make really tiny changes. We can do this in every single column.

![9071](../img/cs231n/winter2016/9071.png)

All the changes add up together.

![9072](../img/cs231n/winter2016/9072.png)

We blew out the probability. This was just a small number of inputs.

In images -> We can nudge 150,528 pixels in a really small way, and we can get any class we want.

---

We can do this in linear classifiers (it has nothing to do with deep learning or ConvNets).

![9073](../img/cs231n/winter2016/9073.png)

We can make templates for all datasets.

We can mix a bit of Goldfish weights -> 100% goldfish.

![9074](../img/cs231n/winter2016/9074.png)

We can do this with original images altered too.

![9075](../img/cs231n/winter2016/9075.png)

We can make a goldfish into a daisy.

![9076](../img/cs231n/winter2016/9076.png)

This has nothing to do with ConvNets or Deep Learning.

We can do this in any other modality, like speech recognition too.

![9077](../img/cs231n/winter2016/9077.png)

To prevent this, you can train the ConvNet with parts of the image or augmentations to make it a little stronger.

You can try to train the ConvNet with training data and adversarial examples (as negative scores), but you can always find new adversarial examples.

You can change the classifier, and this kinda works, but now your classifiers do not really work as they used to.

#### Implications

We saw that backpropping to the image can be used for **understanding**, **segmenting**, **inverting**, **fun**, *and* **confusion**.

![9078](../img/cs231n/winter2016/9078.png)

We will go into RNNs.

![9079](../img/cs231n/winter2016/9079.png)

What are these?

![9080](../img/cs231n/winter2016/9080.png)

What are these?

![9081](../img/cs231n/winter2016/9081.png)

What are these?

![9082](../img/cs231n/winter2016/9082.png)

Done with lecture 9!
