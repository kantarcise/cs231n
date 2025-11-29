Part of [CS231n Winter 2016](../index.md)

---
## Lecture 15: Course Recap and Guest Lecture by Jeff Dean

---

There is no recorded lecture for this session. Instead, we have a recap of the course followed by notes from Jeff Dean's guest lecture.

![15001](../img/cs231n/winter2016/15001.png)

### Course Recap

We started by defining **Score Functions** to map pixels to class scores.

![15002](../img/cs231n/winter2016/15002.png)

Then we introduced **Loss Functions** to measure how good our predictions are.

![15003](../img/cs231n/winter2016/15003.png)

We learned how to optimize these functions using **Gradient Descent** and backpropagation.

![15004](../img/cs231n/winter2016/15004.png)

We looked at more powerful linear classifiers and score functions.

![15005](../img/cs231n/winter2016/15005.png)

We saw that bigger models generally gave us better results.

![15006](../img/cs231n/winter2016/15006.png)

We dove deep into the **Learning Process**, understanding activation functions, initialization, and regularization.

![15007](../img/cs231n/winter2016/15007.png)

We explored **Convolutional Neural Networks (ConvNets)**, the core of modern computer vision.

![15008](../img/cs231n/winter2016/15008.png)

We explored them further, looking at standard architectures like AlexNet, VGG, and GoogLeNet.

![15009](../img/cs231n/winter2016/15009.png)

We discussed their potential downfalls and how to visualize what they learn.

![15010](../img/cs231n/winter2016/15010.png)

We learned about **Style Transfer** and generating art with neural nets.

![15011](../img/cs231n/winter2016/15011.png)

We discovered architectural tricks and newer models like ResNets.

![15012](../img/cs231n/winter2016/15012.png)

We discussed how to make them work in practice, covering libraries like Caffe, Torch, and TensorFlow.

![15013](../img/cs231n/winter2016/15013.png)

We looked at hardware bottlenecks and implementation details.

![15014](../img/cs231n/winter2016/15014.png)

We saw that there are many ways to approach classification and detection.

![15015](../img/cs231n/winter2016/15015.png)

We learned about **Recurrent Neural Networks (RNNs)** and **LSTMs** for sequence modeling.

![15016](../img/cs231n/winter2016/15016.png)

We tackled complex tasks like Image Captioning.

![15017](../img/cs231n/winter2016/15017.png)

You are now ready.

![15018](../img/cs231n/winter2016/15018.png)

Go forth and conquer.

![15019](../img/cs231n/winter2016/15019.png)

The future of computer vision is bright.

![15020](../img/cs231n/winter2016/15020.png)

The End.

![15021](../img/cs231n/winter2016/15021.png)

Thank you all!

![15022](../img/cs231n/winter2016/15022.png)

---
### Guest Lecture: Jeff Dean

Jeff Dean gave a guest lecture on large-scale deep learning at Google.

**Background**:

-   Andrew Ng spent a week at Google in 2011, which kickstarted the Google Brain project.
-   **Google Brain** started in 2011.

**Research Areas**:

-   Speech Recognition
-   Computer Vision (Images, Videos)
-   Robotics
-   Language Understanding (NLP, Translation)
-   Optimization Algorithms
-   Unsupervised Learning

**Production Applications**:

-   Advertising
-   Search
-   Gmail (Smart Reply, Spam Filtering)
-   Google Photos (Search, Organization)
-   Google Maps (Street View analysis)
-   YouTube (Recommendations, Analysis)
-   Speech Recognition (Android, Home)

**Key Takeaways**:

-   **Performance matters**: Making models run fast is crucial for both research iteration and production deployment.

-   **Scaling**: Scaling both **Data** and **Model Size** yields significant improvements. Large-scale distributed training is essential.

