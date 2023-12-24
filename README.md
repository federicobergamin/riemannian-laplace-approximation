## Riemannian Laplace approximations for Bayesian neural networks


> INFO: I will add all the experiments during the holidays, for now you can find the banana experiment and the regression experiment.

Official repository for the paper "Riemannian Laplace approximations for Bayesian neural networks" accepted at NeurIPS 2023.

We provide the environment with all the dependencies we used to run the experiments in the `geomai.yml` file. To create an environment you can just run the following command:
```
conda env create -f geomai.yml
```
Note that the created environment will be called `geomai`.

The next step is to activate the environment:
```
conda activate geomai
```

> NOTE: this code was used for research purposes and the implementation is not really optimized. We are relying on a off-the shelf `scipy` ODEs solver running on CPU, therefore there is a lot of overhead by moving data from CPU and GPU at every step of the solver. In addition to that, we think that re-implementing everything using Jax should provide benefits in terms of speed.

## Banana dataset experiments
In this section we present how to reproduce the results on the Banana dataset. To run vanilla Laplace and our Vanilla Riemannian Laplace method you can just run:
```
# if you want to optimize the prior
python banana_experiment.py -s 0 -str full -sub all  -samp 50 -optim sgd -opt_prior True

# without prior optimization
python banana_experiment.py -s 0 -str full -sub all  -samp 50 -optim sgd 
```
This will use 50 samples from the posterior to generate the confidence plots.

If instead you are interested in generating results using linearized LA or our linearized manifold you can run the following command:
```
# if you want to optimize the prior
python banana_experiment.py -s 0 -str full -sub all  -samp 50 -optim sgd -opt_prior True -lin True

# without prior optimization
python banana_experiment.py -s 0 -str full -sub all  -samp 50 -optim sgd -lin True

# if you want to solve exponential maps using different 
# subset of the training set
python banana_experiment.py -s 0 -str full -sub all  -samp 50 -optim sgd -lin True -batches True
```

## Regression examples on the Snelson dataset

Since in the regression example the Hessian is really ill-defined, we just consider the case when we optimize the prior which alleviate the problem. In case you run it without prior optimization solving the exponential maps can take forever. That's still an open problem from my side to make it run faster in case we do not optimize the prior.

```
python regression_experiment.py -s 0 -str full -sub all  -samp 50 -optim sgd -opt_prior True -opt_sigma True -small True 
```





