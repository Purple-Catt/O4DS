# Optimization for Data Science project

($M$) is so-called _extreme learning_, i.e., a neural network with one hidden layer, $z=W_2\sigma(W_1x)$, where the weight matrix for the hidden layer $W_1$ is a fixed random matrix, $\sigma(\cdot)$ is an elementwise activation function of your choice, and the output weight matrix $W_2$ is chosen by minimising the Mean Square Error of the produced output $f(z)=\Vert W_2\sigma(W_1x)-y\Vert_2^2$ plus a $L_1$ (LASSO) regularization term. The project will have to involve the solution of at least medium-sized instances, i.e., at least 10000 weights and 1000 inputs.  
  
($A_1$) is a standard [momentum descend (heavy ball) approach](https://www.cs.toronto.edu/~fritz/absps/momentum.pdf).  
  
($A_2$) is an algorithm of the class of [deflected subgradient methods](https://pages.di.unipi.it/frangio/abstracts.html#MPC16).  
  
No off-the-shelf solvers allowed.  

  [Link to the Overleaf report](https://www.overleaf.com/1116228861kkcrxpzkkkbd#1d325b)

## Python code
The whole neural network, as well as the training and optimization iterative processes, has been developed "from scratch". This choice allows a better interpretability of the code than the, even partial, use of a high level library (e.g. *Keras* or *PyTorch*).

One of the strenghts of the extreme learning machine (*breviter* ELM) is its extremely fast learning speed. This leads us to implement an **online learning** algorithm, given also the simplicity of its implementation than a **mini-batch** one.

The Python files contain all the necessary parts for the project and their names are self-explanatory of their purpose. For further information about how classes, methods and functions works please refer to the high-detailed descriptions in the code.
