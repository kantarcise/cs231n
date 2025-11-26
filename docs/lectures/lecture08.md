Part of [CS231n Winter 2016](../index.md)

---
## Lecture 8: Spatial Localization and Detection

We will give Andrej a little break.

![8001](../img/cs231n/winter2016/8001.png)

Assignment 2 is due on Friday. The midterm is coming up.

We talked about Convolutional Networks. Lower-level features appear early (low levels), and high-level features appear deeper (higher layers).

![8002](../img/cs231n/winter2016/8002.png)

We saw pooling for shrinking spatial dimensions.

![8003](../img/cs231n/winter2016/8003.png)

We saw a bunch of networks and how they are implemented.

![8004](../img/cs231n/winter2016/8004.png)

We saw ResNet and how it changed the game.

![8005](../img/cs231n/winter2016/8005.png)

### Localization and Detection

![8006](../img/cs231n/winter2016/8006.png)

This is another big problem we have.

![8007](../img/cs231n/winter2016/8007.png)

Localization asks: ***where exactly is the class***? Detection is for bounding boxes, while instance segmentation finds contours around objects.

![8008](../img/cs231n/winter2016/8008.png)

We can do both at the same time.

![8009](../img/cs231n/winter2016/8009.png)

ImageNet also has this as a challenge.

![8010](../img/cs231n/winter2016/8010.png)

The class should be correct, and the Intersection over Union (IoU) should be over $0.5$.

![8011](../img/cs231n/winter2016/8011.png)

We can view localization as a regression problem where we generate 4 numbers.

![8012](../img/cs231n/winter2016/8012.png)

This is a simple recipe: take AlexNet or VGG, and download a pretrained model.

Take those fully connected layers that give us class scores and set them aside.

Attach new FC layers to some point in the network. This is the regression head. It's basically the same thing: a couple of FC layers outputting some real-valued numbers.

![8013](../img/cs231n/winter2016/8013.png)

We train this just like we trained the classification network.

#### Loss Function

We train this network exactly the same way.

![8014](../img/cs231n/winter2016/8014.png)

At test time, we use both heads to do classification and localization. We pass an image through the network (where we have trained both the classification and localization heads), get class scores and boxes, and we are done!

![8015](../img/cs231n/winter2016/8015.png)

One detail: there are two main ways to do regression.

Just 4 numbers (class agnostic) or one bounding box per class.

![8016](../img/cs231n/winter2016/8016.png)

Maybe attach it after the last convolutional layer, or after the last Fully Connected layer.

You could just attach it anywhere.

![8017](../img/cs231n/winter2016/8017.png)

We are also interested in localizing multiple objects.

![8018](../img/cs231n/winter2016/8018.png)

This is used in human pose estimation.

There is a fixed number of joints in a human. What is the pose of the person?

We can find all the joints, run the image through a CNN, and find all points for joints $(x,y)$, which gives us a way to find the current pose of the human.

![8019](../img/cs231n/winter2016/8019.png)

Overall, this idea of localization as regression for a fixed number of objects is simple.

![8020](../img/cs231n/winter2016/8020.png)

This will work. But if you want to win competitions, you need to add a little bit of fancy stuff.

![8021](../img/cs231n/winter2016/8021.png)

You still have this dual-headed network, and you will combine predictions.

![8022](../img/cs231n/winter2016/8022.png)

The classification head gives us class scores. The regression head gives us bounding boxes.

![8023](../img/cs231n/winter2016/8023.png)

#### Sliding Window 🎈

If we run the window only on the upper right corner of the image, we will get a class score and a bounding box. We will repeat this on all 4 corners.

![8024](../img/cs231n/winter2016/8024.png)

Corner top right:

![8025](../img/cs231n/winter2016/8025.png)

Corner bottom left:

![8026](../img/cs231n/winter2016/8026.png)

Corner bottom right:

![8027](../img/cs231n/winter2016/8027.png)

#### Bounding Box Merging

![8028](../img/cs231n/winter2016/8028.png)

We only want a single bounding box. We greedily merge boxes (details in the paper).

![8029](../img/cs231n/winter2016/8029.png)

In practice, they used many more than 4 corners. The figure from the paper is below.

![8030](../img/cs231n/winter2016/8030.png)

They finally decide on the best one.

It is pretty expensive to run the network on all crops.

Let's think about networks, convolutions, and Fully Connected layers.

![8031](../img/cs231n/winter2016/8031.png)

#### Fully Convolutional Networks

Now our network consists only of convolutions, pooling, and element-wise operations. We can now run the network on different-sized images.

This will give us a cheap approximation of running the network on different locations.

We previously had 4096 FC units, but we now have $1x1$ convolutional layers.

![8032](../img/cs231n/winter2016/8032.png)

Now, if we are working on $14x14$ input at training time, with no FC, we have $1x1x4096$ convs.

We can share computation in this way. The only extra computation is in the yellow parts.

![8033](../img/cs231n/winter2016/8033.png)

#### OverFeat 🪟

OverFeat uses a different architecture compared to traditional Convolutional Neural Networks (CNNs) by eliminating fully connected layers. Instead, it uses convolutional layers directly connected to the input image. This design choice allows OverFeat to process images of different sizes without the need for resizing or cropping.

By eliminating fully connected layers, OverFeat reduces the number of parameters in the model and avoids the computational overhead associated with processing high-dimensional feature vectors. This design simplifies the model and makes it more efficient, especially for tasks like object detection where processing speed is critical.

The point here isn't really 're-imagining' the FC layer as a convolution step. Instead, it lets you take advantage of efficiencies built into Convolution Operation implementations that aren't present in FC implementations.

Imagine a convolution operation in 1 dimension. Let's say your kernel is 5 numbers. In step 0, I add A+B+C+D+E = A + (B + C + D + E). That costs me 4 add ops. In step 1, I want to add B+C+D+E+F. I can use my cached value and calculate it with (cached_value) + F, which only costs me 1 add op. Efficiencies like this can be scaled to implementations of the convolution operator. However, FC layers operate over the whole input and have no logical place for such caching.

In OverFeat, we're running these operations on 'windows' of the input image. Each window is a lot like a patch of input to a convolutional layer. By transforming the last FC layers into convolution operations, we can treat the whole network as a series of convolution operations and then take advantage of the inherent efficiencies (described above) of convolution operations.

In the classification + localization problem, the 2013 winner used the **OverFeat** method.

VGG used a deeper network, and it improved the results.

ResNet crushed the competition. They used a different localization method: RPNs.

![8034](../img/cs231n/winter2016/8034.png)

Now we move on.

![8035](../img/cs231n/winter2016/8035.png)

### Object Detection 🐱 🐶

![8036](../img/cs231n/winter2016/8036.png)

Can we use regression here too?

![8037](../img/cs231n/winter2016/8037.png)

2 classes.

![8038](../img/cs231n/winter2016/8038.png)

n cats - $nx4$ coordinates.

![8039](../img/cs231n/winter2016/8039.png)

We need something different because we have variable-sized outputs.

![8040](../img/cs231n/winter2016/8040.png)

You have 2 blades: regression and classification. If regression did not work, try classification.

![8041](../img/cs231n/winter2016/8041.png)

Found a cat here!

![8042](../img/cs231n/winter2016/8042.png)

Nothing here.

So what we do is try out different image regions, run a classifier on each one, and this will solve the variable-sized output problem.

#### Exhaustive Search

![8043](../img/cs231n/winter2016/8043.png)

Detection is a really old problem in Computer Vision, which was solved by HoG in 2005 for pedestrians.

![8044](../img/cs231n/winter2016/8044.png)

Do linear classifiers; they are fast.
Run a Linear Classifier at every scale at every position.

- Compute HoG of the entire image at multiple resolutions.
- Score every sub-window of the feature pyramid.
- Apply non-maxima suppression.

![8045](../img/cs231n/winter2016/8045.png)

People took this idea and worked on it. One of the most important paradigms before deep learning was:

#### Deformable Parts Model

We are still working on HoG features, but instead of a simple linear classifier, we have a linear template for objects that varies over spatial positions.

Latent SVM is used here.

It is a more powerful classifier. It still works really fast. We still run it everywhere at every scale.

![8046](../img/cs231n/winter2016/8046.png)

These are just ConvNets.

HOG vs CNN - Histogram is kinda like pooling, edges are like convolutions.

We still have a problem.

![8047](../img/cs231n/winter2016/8047.png)

We use an expensive classifier on certain regions!

#### Region Proposals 🤹

They do not care about classes; they are looking for blob-like structures. They just run FAST.

![8048](../img/cs231n/winter2016/8048.png)

The most famous one is called ***Selective Search***:

Here is more information about [selective search](https://learnopencv.com/selective-search-for-object-detection-cpp-python/). Here is a [Python Package](https://github.com/ChenjieXu/selective_search) for it.

You start from pixels, merge adjacent pixels if they have similar color and texture, and form connected blob-like features.

After that, you can convert these regions into boxes.

![8049](../img/cs231n/winter2016/8049.png)

There are a lot of different proposal methods. Tip: EdgeBoxes

![8050](../img/cs231n/winter2016/8050.png)

---

#### YOLO Overview

YOLO (You Only Look Once) does not utilize region proposals like some other object detection algorithms such as Faster R-CNN or R-CNN. Instead, YOLO performs object detection by dividing the input image into a grid of cells and predicting bounding boxes and class probabilities directly from each grid cell.

Here's a brief overview of how YOLO works:

- **Grid Division**: YOLO divides the input image into a grid of cells. Each cell is responsible for predicting bounding boxes and class probabilities for the objects contained within it.
- **Bounding Box Prediction**: For each grid cell, YOLO predicts bounding boxes. Each bounding box is represented by a set of parameters: (x, y) coordinates of the box's center relative to the grid cell, width, height, and confidence score. The confidence score indicates how likely the bounding box contains an object and how accurate the box is.
- **Class Prediction**: Along with each bounding box, YOLO predicts class probabilities for the objects present in the bounding box. This is done using softmax activation to estimate the probability of each class for each bounding box.
- **Non-Maximum Suppression (NMS)**: After obtaining bounding boxes and their associated class probabilities, YOLO applies non-maximum suppression to remove redundant or overlapping bounding boxes. This ensures that each object is detected only once with the most confident bounding box.

By directly predicting bounding boxes and class probabilities from grid cells without the need for region proposals, YOLO achieves real-time object detection capabilities. This approach allows YOLO to detect objects efficiently in a single forward pass of the neural network.

---

![8051](../img/cs231n/winter2016/8051.png)

Let's put all of them together.

- We have an input image.
- We will run a region proposal method to get 2000 boxes.
- Crop and warp that image region to some fixed size.
- Run it through a CNN.
- The CNN will have a regression head.

![8052](../img/cs231n/winter2016/8052.png)

Training is a bit complicated. Download a pretrained classification model.

![8053](../img/cs231n/winter2016/8053.png)

We need to add a couple of layers at the end.

We use positive and negative regions from detection images. Initialize a new layer and train again.

![8054](../img/cs231n/winter2016/8054.png)

We want to cache all these features to disk. For every image in your dataset, you run selective search, extract the regions, warp them, and run them through a CNN.

And, you cache the features to disk.

#### Storage Requirements

![8055](../img/cs231n/winter2016/8055.png)

We want to train SVMs to be able to classify different classes based on the features.

![8056](../img/cs231n/winter2016/8056.png)

- You have these image regions.
- You have features for those regions.
- You divide them into positive and negative samples for each class.
- You train binary SVMs.

You do this for every class in your dataset.

![8057](../img/cs231n/winter2016/8057.png)

#### Bounding Box Regression

Your region proposals are not perfect. You might want to make corrections.

If the proposal is too far to the left, you need to regress to this correction vector.

They just do linear regression; you have features and targets, so you just train linear regression.

![8058](../img/cs231n/winter2016/8058.png)

3 datasets are used in practice.

ImageNet has a lot of different images. One object per image.

COCO - a lot more objects per image.

![8059](../img/cs231n/winter2016/8059.png)

`mAP` is the main metric for detection.

![8060](../img/cs231n/winter2016/8060.png)

PASCAL dataset, 2 different versions. Publicly available, so easier to use.

#### Feature Extraction 🍒

![8061](../img/cs231n/winter2016/8061.png)

Pre-CNN vs. Post-CNN is a big improvement in terms of `mAP`.

![8062](../img/cs231n/winter2016/8062.png)

R-CNN has different results for AlexNet, bbox reg + AlexNet, and VGG-16.

![8063](../img/cs231n/winter2016/8063.png)

Features from deeper networks help a lot.

![8064](../img/cs231n/winter2016/8064.png)

R-CNN is pretty slow at test time.

Our SVM and regression are trained offline. The CNN did not have a chance to update.

It's a complex multistage training pipeline.

![8065](../img/cs231n/winter2016/8065.png)

#### Fast R-CNN

- We are just going to swap the order of running a CNN and extracting regions.

Pipeline at test time:

- Take the high-res image.
- Run CNN - get a high-resolution Convolutional feature map.
- We will extract region proposals directly from this convolutional feature map using ROI pooling.
- The convolutional features will be fed to the fully connected layers and classification and regression heads.

![8066](../img/cs231n/winter2016/8066.png)

Computation Sharing

![8067](../img/cs231n/winter2016/8067.png)

Simplified Pipeline

![8068](../img/cs231n/winter2016/8068.png)

**ROI Pooling** 🤔

We have the input image in high resolution, and we have this region proposal coming out of selective search or `edgeboxes`.

We can put this image through convolution and pooling layers fine; those are scale-invariant.

**Problem**: FC layers are expecting low-res conv features.

![8069](../img/cs231n/winter2016/8069.png)

Given a region proposal, we are going to project onto the spatial part of that convolution feature volume.

![8070](../img/cs231n/winter2016/8070.png)

We will divide that feature volume into a grid.

![8071](../img/cs231n/winter2016/8071.png)

We do max pooling.

![8072](../img/cs231n/winter2016/8072.png)

We have taken this region proposal, shared convolutional features, and extracted a fixed-sized output for that region proposal.

**Optimization Tip**: Swapping the order of convolution and wrapping and cropping.

![8073](../img/cs231n/winter2016/8073.png)

We can backpropagate from these regions of interest.

**Joint Training**

Now we can train this thing in a joint way!

![8074](../img/cs231n/winter2016/8074.png)

Much faster! 🐍

![8075](../img/cs231n/winter2016/8075.png)

Test time shows huge improvements.

![8076](../img/cs231n/winter2016/8076.png)

Not a huge improvement in performance, but solid.

![8077](../img/cs231n/winter2016/8077.png)

**Bottleneck**

![8078](../img/cs231n/winter2016/8078.png)

This is still not Real Time.

#### Faster R-CNN

Instead of using some external method, use a network.

**Region Proposal Network**

RPN is trained for region proposal. It looks at the last layer's convolutional features and produces region proposals from the convolutional feature map.

After that, run just like Fast R-CNN.

![8079](../img/cs231n/winter2016/8079.png)

How does it work?

We receive a convolutional feature map as input, coming out of the last layer. RPN is a CNN.

**RPN Convolutions**

Sliding window is a convolution.

- We are doing classification. Is there an object?

- Regression. Regress from this position to an actual region proposal.

The position of the sliding window relative to the feature map tells us where we are in the image.

Regression outputs give us corrections on top of the position on the feature map.

![8080](../img/cs231n/winter2016/8080.png)

It's a little more complicated than that.

**Anchor Boxes**

Taking different sized and shaped anchor boxes and pasting them in the original image at the point corresponding to this point in the feature map.

Every anchor box is associated with a score and a bounding box.

![8081](../img/cs231n/winter2016/8081.png)

In the original paper, training was ugly. Since then, they had some unpublished work where they train this jointly.

They have one big network. In RPN, they have BB regressions and classification loss. They do ROI pooling and do Fast R-CNN.

- We get classification loss on which class it is, and regression loss for correction on top of the region proposal.

**Loss Functions 🐦 🐦 🐦 🐦**

Why not run convolutions on where we want? That is simply external region proposal tactic. We are better of making them with convolutions too because they became the bottleneck.

**Rationale**

RPN is a computational saving.

![8082](../img/cs231n/winter2016/8082.png)

Now we can do object detection all at once. We are not bottlenecked anymore.

![8083](../img/cs231n/winter2016/8083.png)

0.2s is pretty cool.

This is the best object detector in the world (in 2016).

**ResNet Integration**

Fancy stuff for competition:

- **Box refinement**: They do multiple steps for refining the bbox. You saw in the Fast R-CNN framework you are doing that correction on top of your Region Proposal; you can feed that back into the network to re-classify and get another prediction.

- **Context**: In addition to classifying just the image, they get a vector that gives you features on the entire image.

- **Multi-scale testing**: Kinda like OverFeat, they run the thing on different-sized images.

![8084](../img/cs231n/winter2016/8084.png)

In 2013, Deep Learning Detection methods entered the arena.

After 2014, it is all about Deep Learning.

![8085](../img/cs231n/winter2016/8085.png)

#### YOLO

Pose the detection problem directly as a regression problem.

- Divide image into grids.
- Predict B bbox - single score for that bbox.
- Now detection is a regression.

![8086](../img/cs231n/winter2016/8086.png)

It is incredibly fast, but performance is lower.

![8087](../img/cs231n/winter2016/8087.png)

R-CNN: too slow.
Fast R-CNN: requires Matlab.
Faster R-CNN: might be good.
YOLO: is solid.

![8088](../img/cs231n/winter2016/8088.png)

In localization, we are trying to find a fixed number of objects. This is much simpler than detection. We use L2 regression from CNN features to box coordinates.

OverFeat: Regression + efficient sliding window with Fully Connected Layer -> Convolution conversion.

In detection, we are trying to find a varying number of objects. Before CNNs, we used different features and sliding windows. That was costly.

We went from R-CNN to Fast R-CNN to Faster R-CNN.

Deeper is better, with ResNets.

![8089](../img/cs231n/winter2016/8089.png)

Done with lecture 8!
