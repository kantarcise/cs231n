Part of [CS231n Winter 2016](../index.md)

---

## Lecture 5: Training Neural Networks, Part I

Here are some details about the assignments.

![5001](../img/cs231n/winter2016/5001.png)

In this lecture, we transition from the theoretical architecture of neural networks to the practical reality of training them.

We have defined the score function and the loss function, and we know how to compute gradients via backpropagation. Now we must navigate the optimization landscape.

### Project Proposals and Advice

Before beginning the technical content, a few words on course projects.

![5002](../img/cs231n/winter2016/5002.png)

![5003](../img/cs231n/winter2016/5003.png)

One effective strategy is **fine-tuning**. You rarely need to train a network from scratch. Instead, you can take a pre-trained model (trained on a large dataset like ImageNet) and adapt it to your specific problem.

![5004](../img/cs231n/winter2016/5004.png)

You can "chop off" the final classification layer and treat the rest of the network as a fixed feature extractor, training only a new linear classifier on top. Alternatively, you can fine-tune the entire network.

![5005](../img/cs231n/winter2016/5005.png)

There are many pre-trained models available (Caffe Model Zoo, etc.) that you can leverage.

![5006](../img/cs231n/winter2016/5006.png)

A word of caution regarding compute resources:

![5007](../img/cs231n/winter2016/5007.png)

Hyperparameter optimization requires significant computational power. Be mindful of your resource usage, as compute is finite.

![5008](../img/cs231n/winter2016/5008.png)

### Training Overview

We are now at the stage where we loop through the training process:

![5009](../img/cs231n/winter2016/5009.png)

1.  **Sample** a batch of data.
2.  **Forward** prop to compute loss.
3.  **Backprop** to compute gradients.
4.  **Update** parameters.

This is an optimization problem.

![5010](../img/cs231n/winter2016/5010.png)

Neural networks can be incredibly large and complex.

![5011](../img/cs231n/winter2016/5011.png)

However, the complexity is managed by the chain rule. We simply need to implement the `forward` and `backward` API for each module.

![5012](../img/cs231n/winter2016/5012.png)

![5013](../img/cs231n/winter2016/5013.png)

For example, a simple multiplication gate:

![5014](../img/cs231n/winter2016/5014.png)

We can think of these as LEGO blocks that we stack together.

![5015](../img/cs231n/winter2016/5015.png)

We have seen activation functions before, which introduce non-linearity.

![5016](../img/cs231n/winter2016/5016.png)

And we have discussed the loose inspiration from biological neurons.

![5017](../img/cs231n/winter2016/5017.png)

In a fully connected network, the layers with learnable weights (Fully Connected layers) are interleaved with activation functions.

![5018](../img/cs231n/winter2016/5018.png)

### History and Context

It is helpful to zoom out and look at the history of this field.

![5019](../img/cs231n/winter2016/5019.png)

**1957: The Perceptron (Rosenblatt)**: Early implementations were built with hardware circuits.

![5020](../img/cs231n/winter2016/5020.png)

The activation function was a binary step function. Since this is not differentiable, backpropagation as we know it was not possible. They used simple update rules.

**1960: Adaline/Madaline (Widrow & Hoff)**:
Researchers started stacking these units (Multilayer Perceptron).

![5021](../img/cs231n/winter2016/5021.png)

However, without a way to train the hidden layers effectively, progress stalled.

**1986: Backpropagation (Rumelhart, Hinton, Williams)**:
The field was reignited by the derivation of backpropagation, allowing training of multi-layer networks.

![5022](../img/cs231n/winter2016/5022.png)

Despite the excitement, training deep networks proved difficult. Gradients would vanish or explode, and training would get stuck.

**2006: Deep Learning & RBMs (Hinton, Salakhutdinov)**:
A breakthrough came with ***Deep Learning.*** The key idea was unsupervised pre-training using Restricted Boltzmann Machines (RBMs).

![5023](../img/cs231n/winter2016/5023.png)

You would train the first layer to reconstruct the input, then freeze it and train the second layer, and so on. Finally, you would fine-tune the whole network with backpropagation. This initialization allowed for deeper networks.

**2010-2012: The Explosion**:
By 2010, acoustic modeling (speech recognition) saw huge gains by replacing GMMs with Deep Neural Networks. Then came 2012.

![5024](../img/cs231n/winter2016/5024.png)

AlexNet crushed the ImageNet competition. The field exploded.

Why 2012?

-   Better initialization (no longer needed complex pre-training).

-   Better activation functions (ReLU).

-   More data (ImageNet).

-   Better compute (GPUs).

### Activation Functions

We will now focus on the specific choices we make when designing and training these networks. First: Activation Functions.

![5025](../img/cs231n/winter2016/5025.png)

![5026](../img/cs231n/winter2016/5026.png)

There are many options available.

![5027](../img/cs231n/winter2016/5027.png)

#### Sigmoid
Historically, the sigmoid function was very common. It squashes real-valued inputs to the range [0, 1].

![5028](../img/cs231n/winter2016/5028.png)

However, it has severe problems:

1.  **Vanishing Gradients**: When the neuron is saturated (output close to 0 or 1), the gradient is nearly zero.

![5029](../img/cs231n/winter2016/5029.png)

During backpropagation, this local gradient is multiplied by the upstream gradient. If the local gradient is zero, it "kills" the gradient flow to all previous layers.

![5030](../img/cs231n/winter2016/5030.png)

2.  **Not Zero-Centered**: The output is always positive.

![5031](../img/cs231n/winter2016/5031.png)

If the input $x$ to a neuron is always positive, then the gradients on the weights $w$ will all be either positive or negative (depending on the gradient of the loss).

![5032](../img/cs231n/winter2016/5032.png)

This constrains the updates to be in specific directions (zig-zagging), which is inefficient.

![5033](../img/cs231n/winter2016/5033.png)

Empirically, non-zero-centered data leads to slower convergence. So you want to have things that are zero centered.

3.  **Expensive**: The `exp()` function is computationally expensive compared to simple math operations.

![5034](../img/cs231n/winter2016/5034.png)

When we are training CNN's most of compute time is actually in convolutions and dot products. So we want to make sure that we are using efficient ways to compute these.

Yann Lecun recommended using `tanh()` instead of sigmoids.

#### Tanh
The hyperbolic tangent squashes numbers to [-1, 1].

![5035](../img/cs231n/winter2016/5035.png)

-   **Pros**: It is zero-centered.
-   **Cons**: It still suffers from the vanishing gradient problem when saturated.

#### ReLU (Rectified Linear Unit)
The modern standard: $f(x) = \max(0, x)$.

![5036](../img/cs231n/winter2016/5036.png)

-   **Pros**:
    -   Does not saturate in the positive region.
    -   Computationally very efficient.
    -   Converges much faster (e.g., 6x faster for AlexNet).
-   **Cons**:
    -   Not zero-centered.
    -   **Dead ReLU Problem**: When $x < 0$ gradient dies.

![5037](../img/cs231n/winter2016/5037.png)

If a neuron falls into the negative region, its output is 0 and its gradient is 0. It effectively "dies" and may never recover.

![5038](../img/cs231n/winter2016/5038.png)

In practice, you might find that 10-20% of your network is "dead" if you are not careful.

![5039](../img/cs231n/winter2016/5039.png)

> **Tip**: Initialize biases with a small positive number (e.g., 0.01) to ensure ReLUs start active.

#### Leaky ReLU
Attempts to fix the dead ReLU problem by having a small negative slope (e.g., 0.01) when $x < 0$.

![5040](../img/cs231n/winter2016/5040.png)

#### PReLU (Parametric ReLU)
The slope in the negative region is a learnable parameter $\alpha$. Andrej is not completely sold on them.

![5041](../img/cs231n/winter2016/5041.png)

#### ELU (Exponential Linear Unit)
A recent proposal (Clevert et al., 2015) that has benefits of ReLU but is closer to zero mean.

![5042](../img/cs231n/winter2016/5042.png)

#### Maxout

Proposed by Ian Goodfellow et al. It generalizes ReLU and Leaky ReLU.

$f(x) = \max(w_1^T x + b_1, w_2^T x + b_2)$

![5043](../img/cs231n/winter2016/5043.png)

It has no saturation and no dying ReLU problem, but it doubles the number of parameters per neuron.

![5044](../img/cs231n/winter2016/5044.png)

#### Summary of Activations

![5045](../img/cs231n/winter2016/5045.png)

**Recommendation**:

-   Use **ReLU**. Be careful with your learning rates.

-   Try **Leaky ReLU** or **Maxout**.

-   Try **Tanh** but don't expect much.

-   **Never use Sigmoid**.

![5046](../img/cs231n/winter2016/5046.png)

---

### Data Preprocessing

We generally want our input data to be well-behaved.

![5047](../img/cs231n/winter2016/5047.png)

Standard practice in Machine Learning involves:

1.  **Mean Subtraction**: Center the data around zero.

2.  **Normalization**: Scale the data so each dimension has unit variance.

![5048](../img/cs231n/winter2016/5048.png)

Other techniques like **PCA** and **Whitening** (decorrelating the data) are common in general ML but less common in image processing due to the high dimensionality.

![5049](../img/cs231n/winter2016/5049.png)

**For Images**:

-   Subtract the **mean image** (e.g., AlexNet).

-   Or subtract the **per-channel mean** (e.g., VGGNet).

-   Normalization is usually not strictly necessary because pixel values are already on the same scale (0-255).

![5050](../img/cs231n/winter2016/5050.png)

### Weight Initialization

How do we start the optimization? We cannot initialize all weights to zero.

![5051](../img/cs231n/winter2016/5051.png)

If all weights are zero, every neuron computes the same output and gets the same gradient update. There is no **symmetry breaking**.

![5052](../img/cs231n/winter2016/5052.png)

#### Small Random Numbers

A common first attempt is small random noise: `W = 0.01 * np.random.randn(D, H)`.

![5053](../img/cs231n/winter2016/5053.png)

This works for shallow networks, but fails for deep ones.

Let's look at an experiment with a 10-layer network using Tanh non-linearities.

![5054](../img/cs231n/winter2016/5054.png)

As data flows through the layers, it is multiplied by small numbers (0.01). The activations quickly shrink to zero.

![5055](../img/cs231n/winter2016/5055.png)

Why is this bad? During backpropagation, the gradient on the weights is $X \times dL/df$. If the input $X$ (the activation from the previous layer) is tiny, the gradient will be tiny. The network will not learn.

![5056](../img/cs231n/winter2016/5056.png)

#### Large Random Numbers
What if we use larger weights? `W = 1.0 * np.random.randn(D, H)`.

![5057](../img/cs231n/winter2016/5057.png)

Now the neurons saturate. Tanh outputs become -1 or +1. The gradients become zero. The network does not learn.

#### Xavier Initialization
We want the variance of the input to be the same as the variance of the output.
Glorot and Bengio (2010) derived a formula for this:
`W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)`

![5058](../img/cs231n/winter2016/5058.png)

This keeps the activations well-scaled across many layers.

![5059](../img/cs231n/winter2016/5059.png)

However, this derivation assumes linear activations. If we use **ReLU**, it breaks. ReLU kills half the variance (sets negative values to 0).

![5060](../img/cs231n/winter2016/5060.png)

#### He Initialization
He et al. (2015) corrected this for ReLU by adding a factor of 2.
`W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)`

![5061](../img/cs231n/winter2016/5061.png)

This is the current standard for initializing ReLU networks.

![5062](../img/cs231n/winter2016/5062.png)

---

### Batch Normalization

PS: This is explained in more detail in assignment 2.

Batch Normalization (Ioffe & Szegedy, 2015) is a technique to explicitly force the activations to be unit gaussian throughout the network.

![5063](../img/cs231n/winter2016/5063.png)


**The Idea**:
For each feature dimension, compute the mean and variance over the current mini-batch, then normalize.

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

![5064](../img/cs231n/winter2016/5064.png)

We typically insert this layer after the Fully Connected or Convolutional layer, and *before* the non-linearity.

![5065](../img/cs231n/winter2016/5065.png)

However, we don't want to constrain the network too much. We add learnable parameters $\gamma$ (scale) and $\beta$ (shift) so the network can learn to undo the normalization if it needs to.

$$y = \gamma \hat{x} + \beta$$

**At Test Time**:
We don't use the batch mean/variance. Instead, we use a running average of mean/variance collected during training.

![5066](../img/cs231n/winter2016/5066.png)

**Benefits**:

-   Reduces sensitivity to initialization.

-   Allows higher learning rates.

-   Acts as a regularizer.

![5067](../img/cs231n/winter2016/5067.png)

![5068](../img/cs231n/winter2016/5068.png)

It is good thing to use. But there is a runtime penalty.

**Layer Normalization**:
A related technique is Layer Normalization, which normalizes across the features for a single example, rather than across the batch. This is useful for RNNs or when batch sizes are small.

![batchNorm layerNorm](../img/cs231n/winter2016/batchNorm_layerNorm.png)

---

### Babysitting the Learning Process

Now we look at the practical steps of monitoring training.

![5069](../img/cs231n/winter2016/5069.png)

**Step 1: Preprocessing:**
Zero-center your data.

![5070](../img/cs231n/winter2016/5070.png)

**Step 2: Architecture:**
Choose your architecture (e.g., 2-layer net, 50 hidden neurons).

![5071](../img/cs231n/winter2016/5071.png)

**Step 3: Double Check the Loss:**
Disable regularization. The loss should be around $-\log(1/C)$ where $C$ is the number of classes.
For CIFAR-10 ($C=10$), loss should be $\approx 2.3$.

![5072](../img/cs231n/winter2016/5072.png)

If you add regularization, the loss should go up.

![5073](../img/cs231n/winter2016/5073.png)

**Step 4: Sanity Check (Overfit Small Data):**
Take a tiny subset of data (e.g., 20 examples). Turn off regularization. Train.
You should be able to get 100% accuracy and loss of 0.

![5074](../img/cs231n/winter2016/5074.png)

![5075](../img/cs231n/winter2016/5075.png)

If you can't overfit a small dataset, your model is broken.

**Step 5: Find Learning Rate:**
Now use the full dataset (with small regularization). Start with a small learning rate.

![5076](../img/cs231n/winter2016/5076.png)

If the loss doesn't go down, the learning rate is too low.

![5077](../img/cs231n/winter2016/5077.png)

![5078](../img/cs231n/winter2016/5078.png)

Notice that loss barely changes, but accuracy jumps? This is because weights are shifting slightly to make correct scores just barely higher.

![5079](../img/cs231n/winter2016/5079.png)

If the learning rate is too high, the loss explodes (NaN).

![5080](../img/cs231n/winter2016/5080.png)

![5081](../img/cs231n/winter2016/5081.png)

You want to find a learning rate that is "just right" (roughly in the range [$1e^{-3}$, $1e^{-5}$]).

![5082](../img/cs231n/winter2016/5082.png)

![5083](../img/cs231n/winter2016/5083.png)

---

### Hyperparameter Optimization

We need to find the best hyperparameters (Learning Rate, Regularization, Dropout, etc.).

![5084](../img/cs231n/winter2016/5084.png)

**Strategy: Coarse to Fine**
First, search a wide range for a few epochs.

![5085](../img/cs231n/winter2016/5085.png)

**Tip**: Optimize in **Log Space**.
Learning rates and regularization strengths are multiplicative. Sample exponents uniformly from a range.
`10 ** uniform(-3, -6)`

![5086](../img/cs231n/winter2016/5086.png)

Once you find a good region, narrow the search and run for longer.

![5087](../img/cs231n/winter2016/5087.png)

**Random Search vs. Grid Search**
Always use Random Search.

![5088](../img/cs231n/winter2016/5088.png)

Grid search is inefficient because some hyperparameters are more important than others. Random search explores more unique values for the important parameters.

![5089](../img/cs231n/winter2016/5089.png)

**Visualizing Results**
Plot your results.

![5090](../img/cs231n/winter2016/5090.png)

You cannot spray and pray :).

![5091](../img/cs231n/winter2016/5091.png)

If your best results are on the edge of your search range, you need to shift the range!

![5092](../img/cs231n/winter2016/5092.png)

![5093](../img/cs231n/winter2016/5093.png)

![5094](../img/cs231n/winter2016/5094.png)

### Evaluation

Monitor your loss curves.

![5095](../img/cs231n/winter2016/5095.png)

(Check out lossfunctions.tumblr.com for examples of loss curves).

![5096](../img/cs231n/winter2016/5096.png)

Monitor the gap between training and validation accuracy.

-   Big gap = Overfitting (increase regularization).

-   No gap = Underfitting (increase model capacity).

![5097](../img/cs231n/winter2016/5097.png)

**Weight:Update Ratio**
Track the ratio of the update magnitude to the weight magnitude. It should be around $1e^{-3}$.

![5098](../img/cs231n/winter2016/5098.png)

### Summary

![5099](../img/cs231n/winter2016/5099.png)

We have covered:

- Activation Functions (use ReLU).

- Data Preprocessing (zero-center).

- Weight Initialization (use Xavier/He).

- Batch Normalization (use it).

- Hyperparameter Optimization (random search in log space).

![5100](../img/cs231n/winter2016/5100.png)

In the next lecture, we will continue with parameter updates and more advanced training techniques.
