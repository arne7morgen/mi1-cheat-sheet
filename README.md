
# Machine intelligence 1

Artificial Neural Networks (ANNs)
===============
* simple elements
* massively parallel systems
* low precision (individual elements) & robustness (system)
* distributed representation of information
* __no separation between data and program__

Inductive learning (Learning from observations)
* data driven, adaptive systems
* learning & self-organization vs. deduction & programming
* often seen as a plus: biologically inspired learning rules

## Perceptron (Connectionist Neurons)
### Basics
Connectionist neurons can be modeled as a **linear filter**

![](images/Auswahl_2016-03-12_001.png)

![](images/Auswahl_2016-03-12_002.png)  

Where the **input vector** $x$ can be any source of input, from feature sets to images encoded as vectors.  
The **transfer function** $f$ may be logistic or hyperbolic tangent.  

![](images/Auswahl_2016-03-12_005.png)

The **threshold** $\theta$ *shifts* the data points $x$ to the left or right.  
To incorporate $\theta$, $x$ can be extended by a constant value (1 is mostly chosen) with weight $w_{i0} = - \theta$  
Since $\theta$ is now part of the weight vector, it will be learned automatically.  
This is also called the **Bias**.

![](images/Auswahl_2016-03-12_003.png)

Why does this work? In $h_i$ the $\theta$ is constant. Lets say $x = (x_1, \cdots, x_N)^\top$. Then we write the following:  

$$h_i = \sum_j w_{ij}x_j - \theta_i = \sum_j [w_{ij}x_j] - \theta_i $$

If we now extend $x$ by $x_0 = 1$ and do the same with **w** and initialize a random $-\theta = w_{i0}$ it follows that

$$h_i = \sum_{j=0} w_{ij}x_j = 1\cdot(-\theta) + \sum_{j=1} w_{ij}x_j$$

Which is exactly the same as above. Another intuition can be found [here](http://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks)

### Evaluation
Interpretation of the transfer function, data points and weight vector:

![](images/Auswahl_2016-03-12_004.png)

The evaluated $y$ value is the binary result value
## Multilayered Perceptron
* Recurrent Networks
  * A graph of perceptrons that contain cycles
  * can model dynamic systems, sequence processing
* Feedforward Networks
  * A graph with single start and end node that does not contain any cycles
  * association between variables an prediction of attributes (regression, classification).

![](images/Auswahl_2016-03-14_001.png)

**MLPs are universal function approximators**: Even with simple nonlinear functions, MLPs provide a model class with powerful computational capabilities

#### Error functions

For learning, the results of classification or regression need to be evaluated.

![](images/Auswahl_2016-03-14_002.png)

**Model selection**: Find model (parameters) such that: $E^T = \min$. **(Empirical risk minimization (ERM))**

Example error functions:

![](images/Auswahl_2016-03-14_003.png)

Most commonly used function is the quadratic error function $e_{(y_T,x)} = \frac{1}{2}(y_T - y_{(x)})^2$ with $y_T$ being the **true value** of the attribute and $y_{(x)}$ being the **predicted value** of the attribute.

The problem with $E^T$ is, that we only calculate the error among the data points we know. How can we know that the Error on previously unknown points is also low ? (**generalization error**)

![](images/Auswahl_2016-03-14_004.png)

Problem:
* $P_{(y_T,x)}$ is normally unknown
* If it was known, we could evaluate all data points using it directly and would have no need to learn anything. Is is our goal here to come as close to $P$ as possible.
* To find $E^G$, cross-validation can be used. Using this method, we simulate portions of $P$ in every run.
* Hence, *Mathematical analysis*: When does "$E^T = \min$" imply "$E^G$ is small
(enough)"? $\to$ statistical learning theory, ch. 2 of the script.

Think of the classical problem of over and underfitting.

![](images/Auswahl_2016-03-14_005.png)

* left: average $E^T$, probably low $E^G$ $\to$ good predictor
* right: small $E^T$, probably high $E^G$ $\to$ bad predictor

### Gradient Methods

We want to minimize $E^T$ which is a function of the weight vector **w**

![](images/Auswahl_2016-03-14_006.png)

So we basically need to know if we need to increase od decrease **w** such that $E^T$ goes down.

![](images/Auswahl_2016-03-14_007.png)

The Problem with this is, that **w** is commonly not a scalar value. We need to calculate every dimension of the feature space ($\alpha$) separately

![](images/Auswahl_2016-03-14_008.png)

The learning step $\eta$ is defined to be negative because the gradient points into the exact opposite direction.

How can we calculate the gradient if we don't know $e^{\alpha}_{[w]}$ in advance?

![](images/Auswahl_2016-03-14_009.png)

By simply inserting $\partial y$ we can split up the fraction. The right side only depends on the model class. Hence, it is constant for the model evaluation and we can ignore it because the exact value is not of importance, only its direction.  
On the left, we hopefully chose an error function that can be easily derived by $y$ (like the quadratic error function).

![](images/Auswahl_2016-03-14_010.png)

Therefore, for every error function, its derivative regarding $y$ needs to be provided in order to perform gradient descent.

#### Problems
Obviously there are the typical gradient descent problems.
* choice of $\eta$ is critical for optimization to stop somewhere in time and for finding any good approximation of the optimal result.
* For non convex error functions we may end up in local minima.

### Backpropagation


## Generalization

### Bias and Variance

### Regularization
Regularization fdor Neural Networks
### Cross Validation

## Radial Basis Functions

------------------------------
Support Vector Machines
=======================
Bock:

keinen Bock: Arne
## Elements of Statistical Learing Theory

### Bonferroni

### Nor Free Lunch

### Curse of Dimensionality

$a_3$

$a*4$

## Support Vector Machines

### Structural Risk Mimimazation

### Kernel

### Primal Problem

### Dual Problem

### C-SVM

### Regularization for SVM

### Non-Linear Programming
----------------
Probablilistic Methods
==========================

## Uncertainty and Inference
How do we model the/a world?
### First order logic:

rules that can be used for deductive reasoning. But the first oder logic fails in many situations:
* complete set of antecendents and consequences too large
* no complete theory for domains
* incomplete observations
* stochastic enivironments

### Degrees of Belief

* $P(H): H \rightarrow [0,1] $  
* $P(H) = 0$ H is false
* $P(H) = 1$ H is true
* $0<P(H)<1$ quantifies the **degrees of Belief**

$P(H)$ obeys the laws of probability theory, but there is no justification via repeated observations and stochastic outcome.

### Describing the world with Degrees of Belief
#### Random Variables
* *Random variable:* A part of the world whose status is initially unknown
* *Domain of a random variable:* values the variable can take on

**Examples:**
* Boolean variables
  * variable *cavity*; domain: {true, false}  
* Discrete ordinal variables
  * variable *weather*; domain: {Sunny, rainy, cloudy, snow}  
* Continuous variables
  + variable *temperature*; domain: $R_0^+$

#### Atomic events
A descriptio of the modeled worls is a complete set of the random variables. An **Atomic event** is a complete specification of the random variables that describe the world.
+ atomic events are mutually exclusive
+ set of atomic events must be exhaustive

#### Prior
The **prior** is a specification of the knowledge about the world, without any other information.

**Example:**
+ $P(weather = sunny) = 0.3$

#### Conditional probabilities
The **conditional probabilities** specifies the knowledge of the world, given a set of obervations (evidence).
![](images/bayesInf-01.png)
![](images/bayesInf-02.png)

$P(C|t)$: degree of belive in $C$ given *all* we knoe is $t$.

#### Inference using Joint Probabilities

### Conditional Independence

### Bayes Theorem


## Baysian Networks

### Graphs

#### Cliques

#### Seperators

#### Knowledge Graphs

#### Markov Blanked

### Belief Propagation

## Bayes and Neural Networks
