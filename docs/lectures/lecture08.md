Part of [CS231n Winter 2016](../index.md)

---
# From Justin Johnson

Let's give him the benefit of the doubt.

---
# Spatial Localization and Detection

We will give Andrej a little bit of break.

![8001](../img/cs231n/winter2016/8001.png)

Assignment 2 - Due on Friday. Midterm coming up.

We talked about Convolutional Networks. Lower level features early (low levels), high level features deeper (higher layers).

![8002](../img/cs231n/winter2016/8002.png)

We saw pooling for shrinking spatially.

![8003](../img/cs231n/winter2016/8003.png)

We saw bunch of networks and how they are implemented.

![8004](../img/cs231n/winter2016/8004.png)

We saw ResNet and how it changed the game.

![8005](../img/cs231n/winter2016/8005.png)
## Localization and Detection is also possible.

![8006](../img/cs231n/winter2016/8006.png)

Another big problem we have.

![8007](../img/cs231n/winter2016/8007.png)

Where exactly is the class - localization. Detection is for bounding boxes, instance segmentation is contours around.

![8008](../img/cs231n/winter2016/8008.png)

Do both at the same time.

![8009](../img/cs231n/winter2016/8009.png)

ImageNet also has this as a challenge.

![8010](../img/cs231n/winter2016/8010.png)

Class should be correct, and IoU should be over $0.5$.

![8011](../img/cs231n/winter2016/8011.png)

We can see localization as a regression problem. We are generating 4 numbers.

![8012](../img/cs231n/winter2016/8012.png)

This is a simple recipe. AlexNet, VGG, download a pretrained model. 

Take those fully connected layers that gives us class scores, set them aside.

Attach new FC layers to some point in the network. The regression head. Basically the same thing. A couple of FC layers outputting some real valued numbers.

![8013](../img/cs231n/winter2016/8013.png)

We train this just like how we trained classification network. 
## Instead of class scores and GT classes, we use L2 loss and GT boxes.

We train this network exactly the same way.

![8014](../img/cs231n/winter2016/8014.png)

At test time we use both heads to do classification and localization. We have an image we have trained the classification head, we have trained the localization heads, we pass it through, we get class scores, we get boxes, we are done!

![8015](../img/cs231n/winter2016/8015.png)

One detail, 2 main ways to regression.

Just 4 numbers or One bounding box per class is also possible.

![8016](../img/cs231n/winter2016/8016.png)

Maybe after last convolutional layer. / Or attach after last Fully Connected layer.

You could just attach it to anywhere?

![8017](../img/cs231n/winter2016/8017.png)

We are also interested in localizing multiple objects.

![8018](../img/cs231n/winter2016/8018.png)

This is used in human pose estimation. 

There is a fixed number of joints in a human. What is the pose of the person? 

We can find all the joints, run it  through CNN, find all points for joints $(x,y)$ that gives us a way to find the current pose of the human.

![8019](../img/cs231n/winter2016/8019.png)

Overall this idea of localization as regression for a fixed number of objects is simple.

![8020](../img/cs231n/winter2016/8020.png)

This will work. But if you want to win competitions, you need to add a little bit fancy stuff.

![8021](../img/cs231n/winter2016/8021.png)

You still have this dual headed network, you will combine predictions.

![8022](../img/cs231n/winter2016/8022.png)

Classification head is giving us class scores. Regression head is giving us bounding boxes.

![8023](../img/cs231n/winter2016/8023.png)
## We can run this in larger images. 🎈

If we run the window on the only upper right corner on the image, we will get a class score and bounding box. We will repeat this on 4 corners.

![8024](../img/cs231n/winter2016/8024.png)

Corner top right:

![8025](../img/cs231n/winter2016/8025.png)

Corner bottom left:

![8026](../img/cs231n/winter2016/8026.png)

Corner bottom right:

![8027](../img/cs231n/winter2016/8027.png)
## we will get 4 bounding boxes after all of that.

![8028](../img/cs231n/winter2016/8028.png)

We only want a single bounding box. Greedily merge boxes (details in paper).

![8029](../img/cs231n/winter2016/8029.png)

In practice they used many more than 4 corners. Figure from paper down below. 

![8030](../img/cs231n/winter2016/8030.png)

They finally decide at the best.

It is pretty expensive to run the network on all crops. 

We think about networks, convolutions and Fully connected layers.

![8031](../img/cs231n/winter2016/8031.png)
## We can convert FC layers into convolutional layers.  🤔

Now our network is only convolutions, pooling and element wise operations. Now we can run the network, on different sized images.

This will give us an cheap approximation of running the network on different locations.

We previously had 4096 FC, but we now has $1x1$ convolutional layers.

![8032](../img/cs231n/winter2016/8032.png)

Now if we are working on $14x14$ input in training time, no FC we have $1x1x496$ convs.

We can share computation in this way. Only extra computation is made in yellow parts.

![8033](../img/cs231n/winter2016/8033.png)
## overfeat ?  🪟

OverFeat uses a different architecture compared to traditional Convolutional Neural Networks (CNNs) by eliminating fully connected layers. Instead, it uses convolutional layers directly connected to the input image. This design choice allows OverFeat to process images of different sizes without the need for resizing or cropping.

By eliminating fully connected layers, OverFeat reduces the number of parameters in the model and avoids the computational overhead associated with processing high-dimensional feature vectors. This design simplifies the model and makes it more efficient, especially for tasks like object detection where processing speed is critical.

The point here isn't really "re-imagining" the FC layer as a convolution step.  Instead, it lets you take advantage of efficiencies built into Convolution Operation implementations that aren't present in FC implementations.

Imagine a convolution operation in 1 dimension.  Let's say you're kernel is 5 numbers.  In step 0, I add A+B+C+D+E = A + (B + C + D + E).  That cost me 4 add ops.  In step 1, I want to add B+C+D+E+F.  I can use my cached value and calculate it with (cached_value) + F.  Which only cost me 1 add op.  Efficiencies like this can be scaled to implementations of the convolution operator.  However, FC layers operate over the whole input and have no logical place for such caching.

In Overfeat, we're running these operations on "windows" of the input image.  Each window is a lot like a patch of input to a convolutional layer.  By transforming the last FC layers into convolution operations, we can treat the whole network as a series of convolution operations and then take advantage of the inherent efficiencies (described above) of convolution operations.

In classification + localization problem, 2013 won with ==overfeat== method.

VGG used deeper network, and it improved the results.

ResNet crushed competition. - They used a different localization method - RPN's.

![8034](../img/cs231n/winter2016/8034.png)

Now we move on.

![8035](../img/cs231n/winter2016/8035.png)
### Object detection! 🐱 🐶

![8036](../img/cs231n/winter2016/8036.png)

Can we use regression here too?

![8037](../img/cs231n/winter2016/8037.png)

2 classes.

![8038](../img/cs231n/winter2016/8038.png)

n cats - $nx4$ cats.

![8039](../img/cs231n/winter2016/8039.png)

We need something different because we have variable sized outputs.

![8040](../img/cs231n/winter2016/8040.png)

You have 2 blades, you have regression and classification. Regression did not work, try classification.

![8041](../img/cs231n/winter2016/8041.png)

Found a cat here!

![8042](../img/cs231n/winter2016/8042.png)

Nothing here. 

So what we do is to try out different image regions, run a classifier on each one and this will solve the variable sized output problem.
### Just try them all. That will be expensive.

![8043](../img/cs231n/winter2016/8043.png)

Detection is really old problem in Computer Vision. Which is solved by HoG in 2005 for pedestrians.

![8044](../img/cs231n/winter2016/8044.png)

Do linear classifiers, they are fast. 
Run Linear Classifier at every scale at every position.

- Compute HoG of all image at multiple resolutions.
- Score every sub-window of the feature pyramid.
- Apply non-maxima suppression.

![8045](../img/cs231n/winter2016/8045.png)

People took this idea and worked on it. One of the most important paradigms before deep learning was:
### Deformable Parts Model ? 

We still are working on HoG features but our model instead of a simple linear classifier, we have linear template for objects and that vary over spatial decisions which will form a little bit.

Latent SVM is in this. 

It is more powerful classifier. Still works really fast. We still run it on everywhere every scale. 

![8046](../img/cs231n/winter2016/8046.png)

These are just Conv Nets LOL.
### Histogram is kinda like pooling, edges is like CNN

We still have a problem.

![8047](../img/cs231n/winter2016/8047.png)

We use expensive classifier on certain regions!
## Region Proposals. 🤹

They do not care about classes, they are looking for blob like structures. They just run FAST.

![8048](../img/cs231n/winter2016/8048.png)

Most famous one is called: Selective Search:

Here is more information about [selective search](https://learnopencv.com/selective-search-for-object-detection-cpp-python/). Here is a [Python Package](https://github.com/ChenjieXu/selective_search) for it.

You start from pixels, you merge adjacent pixels, together if they had similar color and texture, and you form connected blob like features.

After that you can convert these regions into boxes. You end up with bottom right.

![8049](../img/cs231n/winter2016/8049.png)

A lot of different proposal methods.
#### Tip: Just use `EdgeBoxes` if you have to choose one.

![8050](../img/cs231n/winter2016/8050.png)

---
## Tangent - Does ==YOLO== have region proposals? - NO

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

- We have an input image
- WE will run a region proposal methods - get 2000 boxes
- Crop and wrap that image region to some fixes size
- Run it through CNN
- CNN will have regression head - 

![8052](../img/cs231n/winter2016/8052.png)

Training is a bit complicated.

Download pretrained classification model.

![8053](../img/cs231n/winter2016/8053.png)

We need to add couple of layers in the end.

We are using positive - negative regions from detection images. Initialize a new layer and train again.

![8054](../img/cs231n/winter2016/8054.png)

We want to cache all these features to disk. For every image in your dataset, you run selective search, you extract the regions, you warp them, you run it through CNN 

AND 

You cache the features it to disk. 
## LOL - 200GB Disk what? For PASCAL only.

![8055](../img/cs231n/winter2016/8055.png)

We want to train SVM's to be able to classify different classes based on the features.

![8056](../img/cs231n/winter2016/8056.png)

- You have these image regions
- You have features for those regions
- You divide them into positive and negative samples for each class
- You train binary SVM's

You do this for every class in your datasets.

![8057](../img/cs231n/winter2016/8057.png)
### bbox regression

Your region proposals are not perfect. You might want to make a corrections.

In the middle, the proposal too far to the left, you need to regress to this correctness vector.

They just do linear regression, you have features and targets you just train linear regression.

![8058](../img/cs231n/winter2016/8058.png)

3 datasets are used in practice.

Imagenet has a lot of different images. One object per image.

COCO - a lot more objects per image.

![8059](../img/cs231n/winter2016/8059.png)

`mAP` is the main metric on detection.

![8060](../img/cs231n/winter2016/8060.png)

PASCAL dataset, 2 different versions. Publicly available so easier to use.
### Feature Extractor 🍒

![8061](../img/cs231n/winter2016/8061.png)

Pre CNN - post CNN is a big improvement.

![8062](../img/cs231n/winter2016/8062.png)

R-CNN have different results for Alexnet - bbox reg + AlexNet  - VGG-16.

![8063](../img/cs231n/winter2016/8063.png)

Features from deeper network helps a lot.

![8064](../img/cs231n/winter2016/8064.png)

R-CNN is pretty slow at test time.

Our SVM and regression is trained offline. CNN did not had a chance to update.

Complex multistage training pipeline.

![8065](../img/cs231n/winter2016/8065.png)
## Fast R-CNN - swap!

- We are just going to swap the order of running a CNN and extracting regions.

Pipeline at testtime:

- Take the high res image
- Run CNN - get high high resolution Convolutional feature map
- We will extract directly region proposals from this convolutional feature map - using ROI pooling
- The convolutional features will be head to the fully connected layers and classsification and regression heads.

![8066](../img/cs231n/winter2016/8066.png)
### Slved slowness - sharing computation of convolution layers between

![8067](../img/cs231n/winter2016/8067.png)
### Messy training pipeline - we have a simple one!

![8068](../img/cs231n/winter2016/8068.png)
### ROI - Region of Interest Pooling 🤔

We have the input image on high resolution, we have this region proposal that is coming out of selective search or `edgeboxes`.

We can put this image through convolution and pooling layers fine, those are scale invarient.

**Problem** - FC layers are expecting low res conv features.

![8069](../img/cs231n/winter2016/8069.png)

Given a region proposal, We are going to project onto spatial part of that convolution feature volume.

![8070](../img/cs231n/winter2016/8070.png)

We will divide that feature volume into a grid.

![8071](../img/cs231n/winter2016/8071.png)

We do max pooling.

![8072](../img/cs231n/winter2016/8072.png)

We have taken this region proposal we shared convolutional features, we extracted fixed sized output for that region proposal.
### TIP: swapping the order of convolution and wrapping and cropping.

![8073](../img/cs231n/winter2016/8073.png)

We can backpropogate from these regions of interests.
## Now we can train this thing in a joint way!

![8074](../img/cs231n/winter2016/8074.png)

Much faster! 🐍

![8075](../img/cs231n/winter2016/8075.png)

Test time is huge improvements.

![8076](../img/cs231n/winter2016/8076.png)

Not a huge improvement in performance. But solid.

![8077](../img/cs231n/winter2016/8077.png)
### Now the Fast R-CNN is so good that the bottleneck is computing region proposals.

![8078](../img/cs231n/winter2016/8078.png)

This is still not Real Time.
## Faster R-CNN: why not use Convolutions for Region Proposals?

Instead of using some external method, use a network
## Region Proposals Network

RPN is trained for region proposal, look at the last layer convolutional features and produce region proposals from convolutional feature map.

After, run just like Fast R-CNN.

![8079](../img/cs231n/winter2016/8079.png)

How does it work?

We receive as input convolutional feature map, coming out of last layer, RPN is a CNN.
## typo here, we have $3x3$ convolutions.

Sliding window is a convolution. 
- We are doing classification. Is there an object ?
- Regression. Regress from this position to an actual region proposal.

Position of sliding window relative to the feature map tells us where we are in the image.

Regression outputs give us corrections on top of the position on feature map.

![8080](../img/cs231n/winter2016/8080.png)

A little more complicated than that.
## They had anchor boxes. - RPN

Taking different sized and shaped anchor boxes and pasting them in the original image at the point of the image corresponding to this point in the feature map.

Every anchor box is associated with a score and a bounding box.

![8081](../img/cs231n/winter2016/8081.png)

In the original paper, training ugly. Since then they had some unpublished work where they train this jointly. 

They have one big network, in RPN they have BB regressions, they have classification loss, they do ROI pooling and do Fast-R-CNN: 

- we will get classification loss on which class it is, we have regression loss to correction on top of region proposal
## 4 Losses! 🐦 🐦 🐦 🐦

### Why not run convolutions on where we want? That is simply external region proposal tactic. We are better of making them with convolutions too because they became the bottleneck.

RPN is a computational saving.

![8082](../img/cs231n/winter2016/8082.png)

Now we can do object detection all at once. We are not bottle-necked anymore.

![8083](../img/cs231n/winter2016/8083.png)

0.2 is pretty cool.

This is the best object detector in the world.
### 101 ResNet and Faster R-CNN

Fancy stuff for competition: 

- Box refinement: They do multiple steps for refining the bbox. You saw in the Fast R-CNN framework you are doing that correction on top of your Region Proposal, you can feed that back into the network to re-classify and re-get another prediction.

- Context: IN addition to classifying just the image, they get a vector for that gives you whole features on the entire image.

- Multi-scale testing: Kinda like overfeat, they run the thing on different sized images.

![8084](../img/cs231n/winter2016/8084.png)

2013 Deep Learning Detection methods entered the arena.

After 2014, it is all about Deep Learning.

![8085](../img/cs231n/winter2016/8085.png)
### A fun THING - YOLO

Pose the detection problem directly as a regression problem.

- Divide image to grids.
- Predict B bbox - single score for that bbox.
- Now detection is a regression.

![8086](../img/cs231n/winter2016/8086.png)

It is incredibly fast, but performance is down.

![8087](../img/cs231n/winter2016/8087.png)

R-CNN too slow.
Fast R-CNN requires Matlab.
Faster R-CNN might be good.
YOLO is solid.

![8088](../img/cs231n/winter2016/8088.png)

In localization we are trying to find fixed number of objects. Much simpler than detection. L2 regression from CNN features to box coordinates. 

Overfeat: Regression + efficient sliding window with Fully Connected Layer -> Convolution conversation.

Detection, we are trying to find varying number of objects. Before CNN's we used different features and sliding window. That was costly.

We went from R-CNN to Fast R-CNN to Faster R-CNN

Deeper better, with ResNet's.

![8089](../img/cs231n/winter2016/8089.png)

Done !
