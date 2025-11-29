Part of [CS231n Winter 2016](../index.md)

---
## Lecture 12: Deep Learning Libraries

Today we're going to go over the **four major software packages** that people commonly use for deep learning: Caffe, Torch, Theano, and TensorFlow.

![12001](../img/cs231n/winter2016/12001.png)

The final assignment is due on Wednesday.

---

A quick note: if you're using terminal instances for your projects, make sure you are backing up your code and data. We've had some problems where instances crash, and while data is usually recoverable, it can take time.

---
### Framework Overview

Disclaimer: I've mostly worked with Caffe and Torch, so I know the most about them. I'll do my best to give you a good flavor for the others as well.

### Caffe

Caffe sprung out of a paper at Berkeley that tried to re-implement AlexNet and use AlexNet features for other things. Since then, Caffe has grown into a really popular, widely used package, especially for Convolutional Neural Networks.

![12002](../img/cs231n/winter2016/12002.png)

Caffe is mostly written in C++. There are bindings for Python and MATLAB that are super useful.

In general, Caffe is really widely used and is great if you just want to train standard feed-forward vanilla Convolutional Networks.

Caffe is somewhat different from the other frameworks in that you can actually train big powerful models without writing any code yourself. For example, you can train a ResNet ImageNet classification model using Caffe without writing any code, which is pretty amazing.

![12003](../img/cs231n/winter2016/12003.png)

The most important tip when working with Caffe is that the documentation is sometimes out of date. Be courageous, dive in and read the source code yourself.

The C++ code in Caffe is pretty well-structured and easy to understand. If you have doubts about how things work, your best bet is to go on GitHub and read the source code.

![12004](../img/cs231n/winter2016/12004.png)

Caffe is a huge project, but there are really ***four major classes*** you need to know about:

1.  **Blob**: Blobs store all your data, weights, and activations. They are N-dimensional tensors. They actually store two copies: `data` (raw data) and `diff` (gradients). They also have CPU and GPU versions of each.
2.  **Layer**: A layer is a function. It receives input blobs (bottoms) and produces output blobs (tops). In the forward pass, it fills the data of the top blobs. In the backward pass, it computes gradients for the bottom blobs.
3.  **Net**: A Net combines a bunch of layers. It is a directed acyclic graph of layers and is responsible for running the forward and backward methods in the correct order.
4.  **Solver**: The Solver optimizes the network. It runs the net forward and backward, updates parameters, and handles checkpointing. Different update rules (SGD, Adam, RMSProp) are implemented as subclasses.

This gives you an overview of how things fit together: the Net contains Blobs and Layers, and the whole thing is optimized by a Solver.

![12005](../img/cs231n/winter2016/12005.png)

Caffe makes heavy use of ***Protocol Buffers***.

#### Protocol Buffers

Protocol Buffers are like a binary, strongly-typed JSON used widely inside Google for serializing data.

You define a `.proto` file that specifies the fields of your objects.

![12006](../img/cs231n/winter2016/12006.png)

You can serialize instances to human-readable `.prototxt` files.

![12007](../img/cs231n/winter2016/12007.png)

The protobuf compiler generates classes in various languages (C++, Python, Java) to access these data types. Caffe uses protocol buffers to store pretty much everything.

![12008](../img/cs231n/winter2016/12008.png)

Caffe has one giant file called `caffe.proto` that defines all the protocol buffer types used in Caffe. It is thousands of lines long but is the most up-to-date documentation. I encourage you to read it.

![12009](../img/cs231n/winter2016/12009.png)

When working with Caffe, you generally have this four-step process:

1.  **Convert your data**: Use existing binaries to convert images to Caffe's format.
2.  **Define your Net**: Write a `.prototxt` file defining the architecture.
3.  **Define your Solver**: Write a `.prototxt` file defining the optimization parameters.
4.  **Train**: Pass these files to the `caffe train` binary.

This will spit out your trained Caffe model to disk.

![12010](../img/cs231n/winter2016/12010.png)

**Step 1: Convert Data**. Caffe uses **LMDB** by default. If you have images and labels, Caffe has a script to convert them into a giant LMDB file.

![12011](../img/cs231n/winter2016/12011.png)

Caffe has other options (HDF5, reading directly from memory), but LMDB is the easiest to work with in the Caffe ecosystem.

![12012](../img/cs231n/winter2016/12012.png)

**Step 2: Define Net**. You write a big `.prototxt`.

Here is a simple model for logistic regression. It reads data, has a fully connected layer (called `InnerProduct` in Caffe), and a Softmax loss.

![12013](../img/cs231n/winter2016/12013.png)

Every layer typically includes blobs to store data, gradients, and weights. The layer's blobs and the layer itself typically have the same name.

![12014](../img/cs231n/winter2016/12014.png)

You define learning rates for weights and biases directly in the layer definition (`lr_mult`).

![12015](../img/cs231n/winter2016/12015.png)

The number of output classes is specified in `num_output`.

![12016](../img/cs231n/winter2016/12016.png)

To freeze layers, you set the learning rate multipliers to zero.

![12017](../img/cs231n/winter2016/12017.png)

For large models like ResNet or GoogLeNet, the `.prototxt` can get out of hand (ResNet's is almost 7,000 lines). Caffe doesn't support compositionality well, so people often write Python scripts to generate these files.

![12018](../img/cs231n/winter2016/12018.png)

**Fine-tuning**: You typically download an existing `.prototxt` and a `.caffemodel` weights file.

The `.caffemodel` file is a binary containing key-value pairs matching layer names to weights.

![12019](../img/cs231n/winter2016/12019.png)

When you load a model, Caffe tries to match names. If names match, weights are initialized from the file.

![12020](../img/cs231n/winter2016/12020.png)

If names don't match, layers are initialized from scratch. This is how you reinitialize the output layer for a new task (e.g., changing from 1000 ImageNet classes to 10 classes): just change the layer name in the `.prototxt`.

![12021](../img/cs231n/winter2016/12021.png)

**Step 3: Define Solver**. This is another `.prototxt` defining learning rate, decay, checkpointing, etc.

![12022](../img/cs231n/winter2016/12022.png)

**Step 4: Train**. Call the `caffe train` binary with your solver and weights.

![12023](../img/cs231n/winter2016/12023.png)

You can specify which GPU to run on, or use `-1` for CPU only.

![12024](../img/cs231n/winter2016/12024.png)

Caffe supports data parallelism. You can pass multiple GPU IDs or `-gpu all` to automatically split mini-batches across GPUs.

![12025](../img/cs231n/winter2016/12025.png)

**Model Zoo**: Caffe has a great Model Zoo where you can download pre-trained models (AlexNet, VGG, ResNet, etc.). This is a really strong point of Caffe.

![12026](../img/cs231n/winter2016/12026.png)

**Python Interface**: Caffe has a Python interface (PyCaffe). It's useful but documentation is sparse; read the code.

![12027](../img/cs231n/winter2016/12027.png)

The Python interface lets you do complex initialization, run networks forward/backward with numpy arrays (great for DeepDream or visualization), and **extract features**.

You can also define layers in Python, but they will be CPU-only, which incurs communication overhead.

![12028](../img/cs231n/winter2016/12028.png)

**Caffe Pros and Cons**:

-   **Pros**: Good for feed-forward networks, no code required, great Model Zoo, Python interface for feature extraction.
-   **Cons**: Cumbersome for big networks (ResNet) or RNNs, writing new layers requires C++ and CUDA.

**Can you do cross-validation?**
In the train/val `.prototxt`, you can define training and testing phases.

![12029](../img/cs231n/winter2016/12029.png)

### Torch

Torch is my personal favorite. It is from NYU, written in C and Lua, and used a lot at Facebook and DeepMind.

![12030](../img/cs231n/winter2016/12030.png)

![12031](../img/cs231n/winter2016/12031.png)

The big thing that freaks people out is **Lua**.

Lua is a high-level scripting language, similar to JavaScript. It uses Just-In-Time (JIT) compilation, so loops are actually fast (unlike Python). It uses prototypical inheritance (like JavaScript).

It is 1-indexed, which is annoying, but otherwise easy to pick up.

![12032](../img/cs231n/winter2016/12032.png)

The main idea behind Torch is the **Tensor** class. It is very similar to a numpy array.

![12033](../img/cs231n/winter2016/12033.png)

Here is numpy code for a two-layer ReLU network.

![12034](../img/cs231n/winter2016/12034.png)

Here is the exact same code using Torch tensors in Lua. It's almost a line-by-line translation.

![12035](../img/cs231n/winter2016/12035.png)

In Torch, changing data types is easy (just like casting in numpy).

![12036](../img/cs231n/winter2016/12036.png)

The real reason Torch is great is that **the GPU is just another data type**.

To run on GPU, you import `cutorch` and cast your tensors to `torch.CudaTensor`. Now they live on the GPU.

![12037](../img/cs231n/winter2016/12037.png)

Tensors are like numpy arrays. Documentation is decent.

![12038](../img/cs231n/winter2016/12038.png)

In practice, you use the `nn` (Neural Network) package. This is a wrapper defining a neural network package in terms of tensors.

Here is the same two-layer network using `nn`.

-   Define network as `Sequential`.
-   Add `Linear` and `ReLU` layers.
-   Use `getParameters` to get weights and gradients.
-   Call `forward` and `backward`.
-   Update weights.

![12039](../img/cs231n/winter2016/12039.png)
We have a net.

![12040](../img/cs231n/winter2016/12040.png)

We have weights grad_weights.

![12041](../img/cs231n/winter2016/12041.png)

We have our loss function.

![12042](../img/cs231n/winter2016/12042.png)

We get random data.

![12043](../img/cs231n/winter2016/12043.png)

Run forward.

![12044](../img/cs231n/winter2016/12044.png)

Run backward.

![12045](../img/cs231n/winter2016/12045.png)

Make an update.
![12046](../img/cs231n/winter2016/12046.png)

To run this on GPU, we import `cutorch` and `cunn`, cast our network and loss to CUDA, and cast our data to CUDA.

![12047](../img/cs231n/winter2016/12047.png)

![12048](../img/cs231n/winter2016/12048.png)

Then we just need to cast our network and our loss function to this other data type.

![12049](../img/cs231n/winter2016/12049.png)

In 40 lines of code, we've written a fully connected network that trains on the GPU.

However, vanilla gradient descent is not great. We want to use Adam or RMSProp.

![12050](../img/cs231n/winter2016/12050.png)

Torch gives us the `optim` package.

![12051](../img/cs231n/winter2016/12051.png)

We define a callback function that runs the network forward and backward and returns the loss and gradients. Then we pass this callback to `optim.adam`.

![12052](../img/cs231n/winter2016/12052.png)

In other words, what changes is that we actually need to define this callback function.

So before we were just calling forward and backward exclude explicitly ourselves instead we're going to define this callback function that will run the network forward and backward on data and then return the loss and the gradient.

And now to make an update step on our network we'll actually pass this callback function to this Adam method from the optim package.

![12053](../img/cs231n/winter2016/12053.png)

So this this is maybe a little bit awkward but we you now we can use any kind of update rule using just a couple lines of change from what we had before.

And again this is very easy to add to run on a GPU by just casting everything to CUDA.

![12054](../img/cs231n/winter2016/12054.png)

Caffe implements everything in terms of nets and layers. Caffe has this really hard distinction between a net and the layer.

In torch they don't we don't really draw this distinction everything is just a module. So the entire network is a module, and also each individual layer is a module.

**Modules**: In Torch, everything is a `Module`. The entire network is a module, and each layer is a module.

![12055](../img/cs231n/winter2016/12055.png)

Modules are classes defined in Lua using the tensor API.

Here is the constructor for `Linear`. It sets up weight and bias tensors.

![12056](../img/cs231n/winter2016/12056.png)

Modules implement `updateOutput` (forward) and `updateGradInput` (backward).

Here's the example of the update output for the fully connected layer.

![12057](../img/cs231n/winter2016/12057.png)

![12058](../img/cs231n/winter2016/12058.png)

They also implement `accGradParameters` to compute gradients with respect to weights.

![12059](../img/cs231n/winter2016/12059.png)

Torch has a ton of modules available. Check GitHub for the list.

![12060](../img/cs231n/winter2016/12060.png)
These get updated a lot, so pay attention to the version you're using.

![12061](../img/cs231n/winter2016/12061.png)

It is very easy to write your own modules. You just implement `updateOutput` and `updateGradInput`. You can do whatever arbitrary code you want inside (loops, stochastic things, etc.).

![12062](../img/cs231n/winter2016/12062.png)

But of course using individual layers on their own isn't so useful we need to be able to stitch them together into larger networks.

**Containers**: To stitch layers together, Torch uses containers.

-   `Sequential`: A linear stack.

![12063](../img/cs231n/winter2016/12063.png)

-   `ConcatTable`: Apply different modules to the same input.


![12064](../img/cs231n/winter2016/12064.png)
-   `ParallelTable`: Apply different modules to a list of inputs.

![12065](../img/cs231n/winter2016/12065.png)

![12066](../img/cs231n/winter2016/12066.png)

**nngraph**: For complicated topologies (like DAGs), Torch provides `nngraph`.

Those containers that I just told you should in theory make it possible to implement just about any topology you want.

Torch provides another package called `nngraph`, that lets you hook up things in more complicated topologies pretty easily.

So here's an example if we have three inputs and we want to produce one output, and we want to produce them with this pretty simple update rule.



![12067](../img/cs231n/winter2016/12067.png)

You define symbolic variables and build a graph. `nn.gModule` returns a module implementing this graph.

![12068](../img/cs231n/winter2016/12068.png)

This function is going to build a module using nn graph and then return it.

So here we import the NNgraph package.

This is actually not a tensor this is defining a symbolic variable so this is saying that our our tensor object is going to receive x y and z as inputs and now here we're actually doing symbolic operations on those inputs.

So here we're saying that a we we want to have a point wise addition of x and y store that in a, we want to have point wise multiplication of a and Z and store that in B, and now point wise addition of a and B and store that in C.

These are not actual tensor objects these are now sort of symbolic references that are being used to build up this computational graph in the back end.

And now we can actually return a module here, where we say that our module will have inputs X,Y and Z and outputs C and this `nn.gModule` will actually give us an object conforming to the module API that implements this computation.

So then after we build the module we can construct concrete torch tensors and then feed them into the module that will actually compute the function.

![12069](../img/cs231n/winter2016/12069.png)

**Pre-trained Models**: `loadcaffe` lets you load Caffe models (AlexNet, VGG) into Torch. There are also implementations for GoogLeNet and ResNet.

![12070](../img/cs231n/winter2016/12070.png)

**LuaRocks**: Torch uses `luarocks` (like pip) to manage packages.

![12071](../img/cs231n/winter2016/12071.png)

Useful packages: `cudnn`, `hdf5`, `cjson`, `fbcunn`.

![12072](../img/cs231n/winter2016/12072.png)

**Typical Workflow**:

1.  Pre-processing script (Python) -> HDF5/JSON.
2.  Train script (Lua) reads HDF5, trains model, saves checkpoints.
3.  Evaluate script (Lua) loads checkpoints, generates outputs.

![12073](../img/cs231n/winter2016/12073.png)

Case study on the page.

![12074](../img/cs231n/winter2016/12074.png)

**Torch Pros and Cons**:

-   **Pros**: Flexible, modular, easy to write custom layers, good pre-trained models. Lua is fast (JIT).
-   **Cons**: Lua (unfamiliar to some), less plug-and-play than Caffe (you write code), RNNs can be tricky (sharing weights manually).

**Performance**: Lua is fast because of JIT compilation, similar to JavaScript engines. Python is slower for loops.

![12075](../img/cs231n/winter2016/12075.png)

### Theano

Theano is from Yoshua Bengio's group at the University of Montreal. It is all about **computational graphs**.

![12078](../img/cs231n/winter2016/12078.png)

You define symbolic variables (`x`, `y`, `z`) and compute outputs symbolically.

We're importing Theano and the Theano tensor object.

![12080](../img/cs231n/winter2016/12080.png)

Then we can actually compute to these outputs symbolically so x,y&z are these symbolic things and we can compute a B and C just using these overload operators and that'll be building up this computational graph in the backend.

![12081](../img/cs231n/winter2016/12081.png)

You compile a function using `theano.function`. This is where the magic happens: it optimizes the graph, derives gradients, and generates native code (possibly for GPU).

![12082](../img/cs231n/winter2016/12082.png)

Then you run it on numpy arrays.

![12083](../img/cs231n/winter2016/12083.png)

**Neural Nets in Theano**:

![12084](../img/cs231n/winter2016/12084.png)

Define symbolic variables for inputs, labels, and weights. Here's an example of a simple two-layer ReLU in Theano.

![12085](../img/cs231n/winter2016/12085.png)

The idea is the same, that we're going to declare our inputs, but now instead of just x, y&z we have our inputs in X our labels in Y which are a vector and our two weight matrices `W1` and `W2`.

![12086](../img/cs231n/winter2016/12086.png)

Define forward pass symbolically.

![12087](../img/cs231n/winter2016/12087.png)

Compute loss symbolically.

![12088](../img/cs231n/winter2016/12088.png)

Compile function.
As outputs that will return the loss in a scalar and our classification scores and a vector.

![12089](../img/cs231n/winter2016/12089.png)

**Gradients**: Theano can do symbolic differentiation. `T.grad` computes gradients of the loss with respect to weights.

![12090](../img/cs231n/winter2016/12090.png)

Here we just need to add a couple lines of code to do that.

![12091](../img/cs231n/winter2016/12091.png)

This is the same as before we're defining our symbolic variables for our inputs and our weights.

![12092](../img/cs231n/winter2016/12092.png)
Now the difference is that we actually can do symbolic differentiation here so this `DW1` + `DW2` we're telling Theano that we want those to be the gradients of the loss with respect to those other symbolic variables `W1` and `W2`.

Theano just lets you take arbitrary gradients of any part of the graph with respect to any other part of the graph and now introduce introduce those as new symbolic variables in the graph.

Here in this case we're just going to return those gradients as outputs.
![12093](../img/cs231n/winter2016/12093.png)

You can implement gradient descent by getting gradients and updating weights in a loop (in Python).

![12094](../img/cs231n/winter2016/12094.png)

**Problem**: Updating weights in Python incurs CPU-GPU communication overhead.

![12095](../img/cs231n/winter2016/12095.png)

Every time we call this `f` function and we get back these gradients that's copying the gradients from the GPU back to the CPU and that can be an expensive operation

![12096](../img/cs231n/winter2016/12096.png)

**Shared Variables**: To fix this, use Shared Variables. These live inside the computational graph and persist.

![12097](../img/cs231n/winter2016/12097.png)

![12098](../img/cs231n/winter2016/12098.png)

You define `updates` in the `theano.function` call. This updates the shared variables directly on the GPU every time the function is called.

![12099](../img/cs231n/winter2016/12099.png)

We include this update.

![12100](../img/cs231n/winter2016/12100.png)

**Advanced Theano**: Conditionals, Loops (`scan`), Jacobians, R-operators.

![12101](../img/cs231n/winter2016/12101.png)

Theano has multi-GPU support (experimental).

![12102](../img/cs231n/winter2016/12102.png)

**Lasagne**: A high-level wrapper around Theano. It abstracts away the details.

![12103](../img/cs231n/winter2016/12103.png)

Sweet abstraction.

![12104](../img/cs231n/winter2016/12104.png)

So again we're sort of defining symbolic matrices.

![12105](../img/cs231n/winter2016/12105.png)

And Lasagne now has these layer functions that will automatically set up the shared variables.

![12106](../img/cs231n/winter2016/12106.png)

We can compute the probability in the loss using these convenient things from the Lasagne library.

![12107](../img/cs231n/winter2016/12107.png)

Lasagne writes update rules for you (Adam, Nesterov).

![12108](../img/cs231n/winter2016/12108.png)

We just end up with one of these compiled Theano functions and we use it the same way as before.

![12109](../img/cs231n/winter2016/12109.png)

**Keras**: Even higher-level wrapper. Can use Theano or TensorFlow backend.

![12110](../img/cs231n/winter2016/12110.png)

So here we're having making a sequential container and we're adding a stack of layers to it so this is kind of like torch.

![12111](../img/cs231n/winter2016/12111.png)

And we're having this making this SGD object that is going to actually do updates for us.

![12112](../img/cs231n/winter2016/12112.png)

Problem: Debugging can be hard. Error messages are often cryptic stack traces from the backend.

![12113](../img/cs231n/winter2016/12113.png)

We wrote this kind of simple looking code and Keras but because it's using Theano as a back-end it crapped out and gave us this really confusing error message. 

So that's I think one of the common pain points and failure cases with anything that uses Theano as a back-end.

That debugging can be kind of hard.

![12114](../img/cs231n/winter2016/12114.png)

![12115](../img/cs231n/winter2016/12115.png)

**Pre-trained Models**: Lasagne has a good Model Zoo (AlexNet, VGG, GoogLeNet).

![12116](../img/cs231n/winter2016/12116.png)

![12117](../img/cs231n/winter2016/12117.png)

**Theano Pros and Cons**:

-   **Pros**: Python/Numpy, powerful computational graphs (symbolic gradients), good for RNNs.

-   **Cons**: Raw Theano is ugly, error messages are painful, compile times can be long for big models.

![12118](../img/cs231n/winter2016/12118.png)

### TensorFlow

TensorFlow is from Google. It is shiny, new, and everyone is excited about it.

It is very similar to Theano in that it takes the idea of a computational graph and builds everything on top of it.

![12119](../img/cs231n/winter2016/12119.png)

TensorFlow and Theano are closely linked in concept, which is why Keras can use either as a backend.

One point to make is that TensorFlow is the first of these frameworks designed from the ground up by professional engineers (rather than academic research labs).

![12120](../img/cs231n/winter2016/12120.png)

Here is our favorite two-layer ReLU network in TensorFlow.

![12121](../img/cs231n/winter2016/12121.png)

It is very similar to Theano. We import `tensorflow`.

-   **Placeholders**: Equivalent to Theano's symbolic variables (input nodes).
-   **Variables**: Equivalent to Theano's shared variables (weights).

![12122](../img/cs231n/winter2016/12122.png)

![12123](../img/cs231n/winter2016/12123.png)

We compute the forward pass using library methods that operate symbolically to build the graph.

![12124](../img/cs231n/winter2016/12124.png)

This part looks a bit more like Keras or Lasagne. We use a `GradientDescentOptimizer` and tell it to minimize the loss. We don't explicitly compute gradients or write update rules; the optimizer adds the necessary nodes to the graph.

![12125](../img/cs231n/winter2016/12125.png)

We instantiate numpy arrays for data.

![12126](../img/cs231n/winter2016/12126.png)

**Sessions**: To run code, you wrap it in a `Session`. The session handles the optimization and execution of the graph.

To train, we call `sess.run` and tell it which outputs we want (`train_step`, `loss`) and feed in the data (`feed_dict`).

This is equivalent to calling the compiled function in Theano.

![12127](../img/cs231n/winter2016/12127.png)

**TensorBoard**: One of the coolest things about TensorFlow is TensorBoard, which lets you visualize your network.

![12128](../img/cs231n/winter2016/12128.png)

We add summary nodes to the graph:
-   `scalar_summary` for loss.
-   `histogram_summary` for weights.

![12129](../img/cs231n/winter2016/12129.png)

We merge all summaries and create a `SummaryWriter`.

![12130](../img/cs231n/winter2016/12130.png)

In our loop, we evaluate the `merged` summary object. This computes the summaries, and we write them to disk.

![12131](../img/cs231n/winter2016/12131.png)

Then you start the TensorBoard web server and get beautiful visualizations.

-   Loss curves.
-   Histograms of weights over time.

![12132](../img/cs231n/winter2016/12132.png)

**Graph Visualization**: TensorBoard can also visualize your network structure.

![12133](../img/cs231n/winter2016/12133.png)

You can annotate variables with names and scope computations under namespaces to group them semantically.

![12134](../img/cs231n/winter2016/12134.png)

![12135](../img/cs231n/winter2016/12135.png)

This gives you a visual representation of the computational graph, which is great for debugging.

![12136](../img/cs231n/winter2016/12136.png)

You can click into nodes to see sub-operations.

![12137](../img/cs231n/winter2016/12137.png)

**Distributed Training**: TensorFlow supports data parallelism and model parallelism.

-   **Data Parallelism**: Split mini-batch across devices.
-   **Model Parallelism**: Split the model across devices (useful for large models or multi-layer RNNs).

![12138](../img/cs231n/winter2016/12138.png)

You can also actually do model parallelism in TensorFlow as well that let's you split up the same model and compute different parts of the same model on different devices.

![12139](../img/cs231n/winter2016/12139.png)

TensorFlow is the only framework that supports distributed training across **multiple machines** (not just multiple GPUs on one machine).

*Caveat*: As of today (Winter 2016), the distributed part is not open source yet. Hopefully, it will be released soon.

![12140](../img/cs231n/winter2016/12140.png)

**Pre-trained Models**: Currently lacking. There is an Inception model in an Android demo, but not much else yet.

![12141](../img/cs231n/winter2016/12141.png)

**TensorFlow Pros and Cons**:

-   **Pros**: Python/Numpy, powerful computational graphs, TensorBoard is amazing, data/model parallelism.
-   **Cons**: Slower than others (currently), distributed features not fully open source yet, lack of pre-trained models.

![12142](../img/cs231n/winter2016/12142.png)

### Comparison

Here is a quick overview table comparing the frameworks.

![12143](../img/cs231n/winter2016/12143.png)

**Scenarios**:

1.  **Extract Features (AlexNet/VGG)**: Caffe.

![12144](../img/cs231n/winter2016/12144.png)

![12145](../img/cs231n/winter2016/12145.png)

2.  **Fine-tune AlexNet on new data**: Caffe.

![12146](../img/cs231n/winter2016/12146.png)

![12147](../img/cs231n/winter2016/12147.png)

3.  **Image Captioning with Fine-tuning**: Torch or Lasagne. (Need pre-trained models + RNNs).

![12148](../img/cs231n/winter2016/12148.png)

![12149](../img/cs231n/winter2016/12149.png)

4.  **Semantic Segmentation**: Torch. (Need pre-trained models + custom logic).

![12150](../img/cs231n/winter2016/12150.png)

![12151](../img/cs231n/winter2016/12151.png)

5.  **Object Detection**: Caffe + Python, or Torch. (Complex imperative code).

![12152](../img/cs231n/winter2016/12152.png)

![12153](../img/cs231n/winter2016/12153.png)

6.  **Language Modeling (RNNs)**: Theano or TensorFlow. (No pre-trained models needed, focus on recurrence).

![12154](../img/cs231n/winter2016/12154.png)

![12155](../img/cs231n/winter2016/12155.png)

7.  **Implement Batch Norm (Custom Gradients)**: Torch. (If you want to implement efficient gradients yourself).

![12156](../img/cs231n/winter2016/12156.png)

**Recommendations**:

-   **Feature Extraction / Fine-tuning**: Caffe.
-   **Complex uses of Pre-trained Models**: Lasagne or Torch.
-   **Writing Custom Layers**: Torch.
-   **RNNs / Computational Graphs**: Theano or TensorFlow.
-   **Gigantic Distributed Models**: TensorFlow.

**Speed**: Currently, Neon (Nervana Systems) is fastest (custom assembler). Among the others using cuDNN, speed is roughly the same. TensorFlow is currently a bit slower but should improve.

**Graphing**: Torch has `iTorch` notebooks.

![12157](../img/cs231n/winter2016/12157.png)


Down below are the extras, no time left in the lecture to cover them.

![12158](../img/cs231n/winter2016/12158.png)

![12159](../img/cs231n/winter2016/12159.png)

![12160](../img/cs231n/winter2016/12160.png)

![12161](../img/cs231n/winter2016/12161.png)

![12162](../img/cs231n/winter2016/12162.png)

![12163](../img/cs231n/winter2016/12163.png)

![12164](../img/cs231n/winter2016/12164.png)

![12165](../img/cs231n/winter2016/12165.png)

![12166](../img/cs231n/winter2016/12166.png)

![12167](../img/cs231n/winter2016/12167.png)

![12168](../img/cs231n/winter2016/12168.png)

![12169](../img/cs231n/winter2016/12169.png)

![12170](../img/cs231n/winter2016/12170.png)

![12171](../img/cs231n/winter2016/12171.png)

![12172](../img/cs231n/winter2016/12172.png)

![12173](../img/cs231n/winter2016/12173.png)

Done with lecture 12. 🍒

---

### Extra: How to calculate gradients ? 🍒

In the context of deep learning and neural network training, "parsing out the AST (Abstract Syntax Tree) for calculating gradients" refers to the process of automatically differentiating the computational graph of a neural network to compute the gradients of the model parameters with respect to the loss function.Here's a more detailed explanation:

1. **Computational Graph**: When building a neural network in a deep learning framework like PyTorch or TensorFlow, the network is represented as a computational graph. This graph consists of nodes (representing operations like matrix multiplication, activation functions, etc.) and edges (representing the flow of tensors between operations).

2. **Automatic Differentiation**: To train a neural network using gradient-based optimization methods (like stochastic gradient descent), we need to compute the gradients of the model parameters with respect to the loss function. This process is known as automatic differentiation or backpropagation.

3. **Abstract Syntax Tree (AST)**: The computational graph of a neural network can be represented as an AST, which is a tree-like data structure that captures the structure of the computations performed by the network. Each node in the AST represents an operation, and the edges represent the dependencies between the operations.

4. **Parsing the AST**: To compute the gradients, the deep learning framework needs to "parse" the AST of the computational graph. This involves traversing the AST, identifying the operations, and applying the chain rule of differentiation to compute the gradients of the model parameters with respect to the loss function.

5. **Gradient Calculation**: By parsing the AST, the deep learning framework can automatically compute the gradients of the model parameters with respect to the loss function. This is done by applying the chain rule of differentiation, starting from the output of the network and working backwards through the computational graph.

The ability to automatically differentiate the computational graph and compute the gradients is a key feature of modern deep learning frameworks like PyTorch and TensorFlow. It allows developers to focus on defining the neural network architecture and the loss function, without having to manually derive and implement the gradient calculations.This automatic differentiation process, enabled by parsing the AST of the computational graph, is a crucial component that allows deep learning models to be trained efficiently using gradient-based optimization methods.

---