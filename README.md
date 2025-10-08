# iterative-gaussianization

![Ring](experiments/gifs/ring.gif)


Iterative construction of transport maps that push the standard Gaussian forward to a target distribution specified by an unnormalized density. The transformations alternate between rotations and coordinatewise maps. The rotations are chosen via a score-based PCA procedure, while the coordinatewise maps are obtained by solving mean-field variational inference problems.

## Installation
```
pip install -e .
```

## Run experiments
```
python -m experiments.logistic.run_logistic
python -m experiments.posteriordb_experiment.run_posteriordb --posterior_name normal_mixture
python -m experiments.glmm.run_glmm
python -m experiments.sparselogistic.run_sparse_logistic
python -m experiments.irt2pl.run_irt2pl
```