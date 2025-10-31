Part of [CS231n Winter 2016](../index.md)

---
# Fourth Lecture: Andrej Karpathy

The legend.

---
# Lecture 4: Back-propagation and Neural Networks part 1

![4001](../img/cs231n/winter2016/4001.png)

150 hours left `LOL`. 3rd of those hours you will be unconscious.

![4002](../img/cs231n/winter2016/4002.png)

We have a score function. SVM loss and Regularization Loss.

We want the gradient expression of the loss function with respect to weights. Doing a parameter update, so that we are converging to a low points in the loss function. 

A low loss, means we are making good predictions.

![4003](../img/cs231n/winter2016/4003.png)

Now we also saw that there are two kind of ways to  evaluate the gradient 
## Numerical vs Analytic Gradient.

There's a numerical gradient and this is very easy  to write but it's extremely slow to  evaluate and there's analytic gradient which is which you obtained by using calculus and we'll be going into that in  this lecture.

Numerical is possible. We want the analytic one.

![4004](../img/cs231n/winter2016/4004.png)

You might be tempted to just you  know write out the full loss and just start to take the gradients as you see in your calculus class.

The point I'd like to make is that you should think much more of this in terms of computational graphs instead of just taking thinking of one giant expression that you're going to derive with pen and paper.

![4005](../img/cs231n/winter2016/4005.png)
## think 🧠 in terms of a computational graph

Here we're thinking about these values flow flowing through a ==computational graph== where we have these  operations along circles and they transfer they're basically function pieces that transform your inputs all  the way to the loss function at the end.

So we start off with our data and our parameters as inputs they feed through this computational graph which is just an all these series of functions along the way. 

At the end we get a single  number which is the loss.
## why - because things will get complicated. 📱

![4006](../img/cs231n/winter2016/4006.png)

Convolutional  networks are not even the worst of it

![4007](../img/cs231n/winter2016/4007.png)

basically a differentiable Turing machine - computational graph is huge!

![4008](../img/cs231n/winter2016/4008.png)

A Giant monster of  hundreds of thousands of nodes and  little computational units and so it's impossible to write out you know here's  the loss for the neural Turing machine.. LOL no.
### We're  going to be looking specifically at  computational graphs and how we can derive the gradient on the inputs with respect to the loss function at the very end.

---
# tangent from other course 🍉:

Implementing back propagation by hand is like programming in assembly language. You’ll probably never do it, but it’s important for having a mental model of how everything works.

---
# so let's start off simple

We have this  graph that at the end gives us th is  output negative 12.

![4009](../img/cs231n/winter2016/4009.png)

I've already prefilled what  we'll call the forward pass of this  graph where I set the inputs and then I  compute the outputs.

Now we'd like to do is we'd like to derive the gradients of the expression on the inputs.

![4010](../img/cs231n/winter2016/4010.png)

Intermediate variable Q after sum gate. $q = x + y$ and $f = qz$ has their both derivatives done. 

We've performed the forward paths going from left to right and what we'll do now  is we'll derive the backward pass.
## the backward pass ? 🤔

Gradients in red. Values are in green.

![4011](../img/cs231n/winter2016/4011.png)

As a base case of this recursive procedure we're considering the gradient of F with respect to F. Derivative of f by f is 1. 

![4012](../img/cs231n/winter2016/4012.png)

Now we're going to go backwards through this graph.

gradient of f with respect to z is 3.

![4013](../img/cs231n/winter2016/4013.png)

what does this tell us ? 

What 3 is telling you really (intuitively keep in mind the interpretation of a gradient) is what that's saying is that the influence of Z on the final value is positive and with sort of a force of 3.

So if we increment Z by a small amount H, then the output of the circuit will react by increasing because it's a positive 3, will increase  by $3*H$ so a small change will result in  a positive change on the output.

![4014](../img/cs231n/winter2016/4014.png)

Gradient on Q in this case will be so $DF/DQ$ is said what is that $- 4$.

![4015](../img/cs231n/winter2016/4015.png)

What that's saying is that if Q were to increase then the output of the circuit will decrease, if you increase by H the output of the circuit will decrease by $4*H$

![4016](../img/cs231n/winter2016/4016.png)

we'd like to compute the gradient on F  on Y  with respect to Y.

![4017](../img/cs231n/winter2016/4017.png)

Many ways to do it, just apply chain rule.

![4018](../img/cs231n/winter2016/4018.png)

The chain rule says that if you  would like to derive the gradient of F  on Y then it's equal to $DF / DQ$ times  $DQ / dy$. 

We'll get  - 4 * 1  = $-4$
## this is kind of the the crux of how back propagation works is this is very important to understand.

### we have these two  pieces that we keep multiplying through when we perform this chain rule.

![4019](../img/cs231n/winter2016/4019.png)

Basically what this is saying is that, the influence of Y on the final but the circuit is $-4$ so increasing y should decrease the output of the circuit by $-4 *  the little change$ that you've made.

![4020](../img/cs231n/winter2016/4020.png)

What is X's influence on Q and what q's influence on the end of the circuit and so that ends up being a  chain rule. $-4 * 1 = -4$

So chain rule is kind of giving us this correspondence.
## to generalize a bit from this example:

![4021](../img/cs231n/winter2016/4021.png)
## you are a gate Harry!

You are a gate embedded in a circuit and this is a very large computational graph or circuit. 

You receive some inputs, some particular numbers $X$ and $y$ come in, and you perform some operation on them and compute some output set Z.

This value of Z, goes into computational graph and something happens to it, but you're just a gate  hanging out in a circuit and you're not sure what happens. 

But by the end of the circuit the loss gets computed. 
## That's the forward pass. 

When I get $x$ and $y$ the thing I'd like to point out that during the forward pass if you're this gate and you get your values $x$ and $y$ you compute your output $Z$ and there's  another thing you can compute right away and that is the local gradients on x and y. 

![4022](../img/cs231n/winter2016/4022.png)

I'm just a gate and I know what I'm performing like say addition or multiplication, so I know the influence  that x and y i have on my output value.

![4023](../img/cs231n/winter2016/4023.png)

What happens near the end  so the loss gets computed and now we're  going backwards I'll eventually learn about what is my influence on the final output of the circuit the loss so I'll learn what is $DL / DZ$ in there.

To find $dl/dx$ , use the $dL/dz$ you got and your local gradient in a chain rule!

![4024](../img/cs231n/winter2016/4024.png)

## Chain rule -> `Global gradient ontheoutput` * `localgradient`

Same for $dL/dy$.

![4025](../img/cs231n/winter2016/4025.png)

Chain rule is just this added multiplication where we take our what  I'll called global gradient of this gate  on the output and we change it through  the local gradient.

![4026](../img/cs231n/winter2016/4026.png)

These X's and Y's there  are coming from different gates right so  you end up with recursing this process  through the entire computational circuit  and so these gates just basically  communicate to each other the influence  on the final loss.

So they tell each other, okay if this is a positive gradient that means you're positively  influencing the loss and if it's a negative gradient you are negatively influencing loss.

These just gets all multiplied through the circuit by these local gradients.
# this process is called  back propagation. 💛

It's a way of computing through a recursive application of chain rule through computational graph the influence of every single intermediate value in that graph on the final loss function.

if Z is being influenced in multiple places in the circuit, the backward flows will add but we'll come back to that point.

![4027](../img/cs231n/winter2016/4027.png)

I translated that mathematical  expression into this computational graph  form so we have to recursively from  inside out compute this expression.

![4028](../img/cs231n/winter2016/4028.png)

We're going to do now is we're going to  back propagate through this expression.

We're going to compute what the influence of every single input value is on the output of this expression.

So we have exponentiation and we know for every  little local gate what these local  gradients are right so we can derive that using calculus so $e^X$  derivative is $e^X$

- WE ALWAYS START with $1.00$ - gradient on identity function.
- now we're going to back propagate through this $1 / X$  operation.
- $1 / X$ gate  during the forward pass received input $1.37$ and right away that $1 / X$ gate  could have computed what the local  gradient was, the local gradient was $-1 / x^2$ 
- and now during  back propagation it has to by chain rule  multiply that local gradient by the  gradient of it on the final output of  the circuit which is easy because it  happens to be at the end 
- so we get $-1/ x ^ 2$

![4029](../img/cs231n/winter2016/4029.png)

So the output is $-0.53$. It has negative effect on the output. Which make sense because it is a $1/x$ gate, if we increased x, the value will be smaller.

That's why you're seeing negative gradient.

![4030](../img/cs231n/winter2016/4030.png)

- the next gate in the circuit it's adding a constant of one $+ 1$
- Local gradient is 1. The gradient before is $0.53$

![4031](../img/cs231n/winter2016/4031.png)

- Result is $0.53$

![4032](../img/cs231n/winter2016/4032.png)

- Now we know the gradient just before - $-0.53$
- Local gradient is $e^{-1}$ because input is $-1$ doubt ?

![4033](../img/cs231n/winter2016/4033.png)

- Result is $-0.2$

![4034](../img/cs231n/winter2016/4034.png)

- We have $*-1$ gate. 
- $1 * -1$ gave us $-1$ in the forward pass. Our local gradient it $a$. Result before was $-0.2$

![4035](../img/cs231n/winter2016/4035.png)

- We now have $0.2$

![4036](../img/cs231n/winter2016/4036.png)

- this plus operation has multiple inputs.
- Local gradient to the plus gate is 1 and 1.
- If you just have a  function $X + y$ then for that  function the gradient on either X or Y  is just 1.

![4037](../img/cs231n/winter2016/4037.png)

- what you end up getting  is just $1 * 0.2$
- A plus gate is kind of like a ==gradient distributor== where if something flows in from the top it will just spread out all the all the  gradients equally to all of its children.

We've already received one of the inputs ($w2$) is gradient 0.2 here on the very final output of the circuit.

So this influence has been computed through a series of applications of chain rule along the way.

![4038](../img/cs231n/winter2016/4038.png)

There was another plus gate, so now we're going to back propagate through that (previous) multiply operation.

What will be the gradient for $w0$ and $x0$ ? 

![4039](../img/cs231n/winter2016/4039.png)

- for $w0$ the gradient will be, $-1 * 0.2$
- for $x0$ the gradient will be : $2 * 0.2$

![4040](../img/cs231n/winter2016/4040.png)

Result is not $0.39$ - result is $0.4$. We will skip the next multiplication gate.
### the cost of forward and  backward propagation is roughly equal.

I could have made a different gate.

![4041](../img/cs231n/winter2016/4041.png)

This has a single sigmoid gate for example.

![4042](../img/cs231n/winter2016/4042.png)

If we know the local gradient for the sigmoid, which we can derive, we can use it.

- `Local gradient * gradients = result`

$$(0.73) * (1 - 0.73) * (1.0) = 0.2$$

This understanding helps you debug the networks, for example vanishing gradients.

![4043](../img/cs231n/winter2016/4043.png)

- all addition in your loss function is just distributes gradients
- max gate is gradient router, one of the gradient is highest that came to me, I will pass the max 

Only Z has an influence on the output of this max gate down below (Z > W). 

When $2$ flows into the max gate it gets routed to Z and W gets a $0.0$ gradient because its effect on the circuit is nothing. 

Because when you change it, it doesn't matter when you change it because that is the larger value going through the computational graph.

![4044](../img/cs231n/winter2016/4044.png)

Add up the contributions at the operation.
## Tangent - these are always DAG's , there will not be any loops.

Let's see the implementation.

We need a object to connect the gates. Graphs are great!

It has two main pieces, forward and backward.

When we call forward or backward, we will move all gates forward and backward.

![4045](../img/cs231n/winter2016/4045.png)

This is actual code down below -> a multiply gate:

In the forward pass you just compute whatever.

In the backward pass, you compute, what is our gradient on the final loss.

![4046](../img/cs231n/winter2016/4046.png)

How would we implement $dx$ and $dy$ ? 
## We  have to remember $x$ and $y$ because we will need to use them in backward pass.

# as you are doing forward pass, huge amount of stuff are cached in your memory 

![4047](../img/cs231n/winter2016/4047.png)

You are in charge.
# All a deep learning framework is a bunch of layers. And a very thin computational graph that keeps track of the all layer connectivity 😍

![4048](../img/cs231n/winter2016/4048.png)

We are building towers from Lego blocks.

![4049](../img/cs231n/winter2016/4049.png)

A specific example. 

Just a scaling by a number.

Give the $a$ in the initialization.

Forward and Backward.

![4050](../img/cs231n/winter2016/4050.png)

`Caffe` is a deep learning framework specifically for images. A lot of layers there too.

![4051](../img/cs231n/winter2016/4051.png)

A `Sigmoid` layer. `Caffe` calls tensors blobs. Blob is just a n dimensional array of numbers. 

![4052](../img/cs231n/winter2016/4052.png)

Forward and backward pass, again. Chain rule again.

This is the CPU implementation. There is a separate file which has this run on GPU, which is a CUDA code.
# do we have to go through forward and backward for every update ? 

# yes. When you want to to update you need the gradient. You sample a mini batch, you do a forward, right away you do a backward, now you have initial gradient.

# forward computes the loss
# backward computes the gradient

# update uses the gradients to increment the weights a bit

# This just happens in a loop. 🍎 🍎 🍎

---

We will be working with vectors, not just scalars. 

Nothing changes, just the local gradients are Jacobian matrices.

![4053](../img/cs231n/winter2016/4053.png)
## Jacobian 2D Matrix: What is the influence of every single element in x on every single element of z.

You now have matrix multiplications.

You do not have to form these Jacobians.

![4054](../img/cs231n/winter2016/4054.png)

You did a thresholding.

![4055](../img/cs231n/winter2016/4055.png)

Simple: $4096 x 4096$

![4056](../img/cs231n/winter2016/4056.png)

It actually looks like a almost an identity matrix but some of them are 0 instead of 1 (because they were clamped to zero in forward pass).

![4057](../img/cs231n/winter2016/4057.png)

Usually we do mini batches. 

All the examples in mini batches are computed in parallel. The Jacobian ends up huge, so you do not write them out.

Compute scores, compute margins.

Compute loss do backprop.
## stage your computation.

![4058](../img/cs231n/winter2016/4058.png)

We learned why we need forward/backward.

![4059](../img/cs231n/winter2016/4059.png)
# Neural Networks 🐤

![4060](../img/cs231n/winter2016/4060.png)

This is what we had.

![4061](../img/cs231n/winter2016/4061.png)

A 2 layer network, is just more complex score function.

![4062](../img/cs231n/winter2016/4062.png)

We have a non linearity - an activation function. - `max(0, ...)`
# h - hidden layer - `100` is a hyper parameter.

![4063](../img/cs231n/winter2016/4063.png)

We have a 100 layers in middle, now we can actually classify different colored cars.

We could not do that in Linear SVM.

We now have a multi model car classifier!

![4064](../img/cs231n/winter2016/4064.png)

You just extend this if you want 3-layer Network.

![4065](../img/cs231n/winter2016/4065.png)

Not so complicated to make them.

x input - y output. syn0 - sny1 weights. With logistic regression loss.

![4066](../img/cs231n/winter2016/4066.png)

## Stage your computation and do backpropogation for every single result.

![4067](../img/cs231n/winter2016/4067.png)

Neural networks are not really related to brains.

![4068](../img/cs231n/winter2016/4068.png)

Image search for neurons.

![4069](../img/cs231n/winter2016/4069.png)

This is the real neuron.

It can spike based on the input.

![4070](../img/cs231n/winter2016/4070.png)

The similarity is kinda like down below:

Based on the activation of function, you decide on output.

![4071](../img/cs231n/winter2016/4071.png)

People like to use `sigmoid`. You get the number between 0 - 1. 

![4072](../img/cs231n/winter2016/4072.png)

If we want to implement it:

![4073](../img/cs231n/winter2016/4073.png)

Neural Networks are not really like brains.

![4074](../img/cs231n/winter2016/4074.png)

A lot of options we have. ReLU helps NN's convert faster.

![4075](../img/cs231n/winter2016/4075.png)

We connect layers. 

Input does not have computation so architecture on left has 2 layers. `2 Layer Net`

Input does not have computation so architecture on right has 3 layers. `3 Layer Net`

![4076](../img/cs231n/winter2016/4076.png)

Having them in layers allows us to use vectorized operations. Neurons inside layers can be evaluated in paralel and they all see the same input.

![4077](../img/cs231n/winter2016/4077.png)

As you can see below, short code:

![4078](../img/cs231n/winter2016/4078.png)

When we have more hidden neurons we have in hidden layer, the more it can compute crazy functions.

![4079](../img/cs231n/winter2016/4079.png)

When you decrease the regularization, your network can act advanced.

![4080](../img/cs231n/winter2016/4080.png)
## [Website](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html) for demo.

Your hidden layer literally wraps space in such a way that your last layer, which is just a linear classifier, can separate the input points effectively.

You can see on the right in the image down below, space is warped.

![convnetjs wrappingspace](../img/cs231n/winter2016/convnetjs_wrappingspace.png)

Here is the summary so far.

![4081](../img/cs231n/winter2016/4081.png)
## Is it always better to have more neurons ? 
### Yes. Only Computational constraints.

### The correct way to to constrain your Neural Network to not to overfit your data, is not by making the network smaller, the correct way is to increase your regularization.

## For practical reasons, you will use smaller models.

### Do you regularize each layer equally ? Usually you do, as a simplification.

Depth or width, which is more important ? It depends on the data.

![4082](../img/cs231n/winter2016/4082.png)

![4083](../img/cs231n/winter2016/4083.png)

We had a blast, let's go next.
