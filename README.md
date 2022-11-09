# SPIDER-GDA

Codes for "Faster Stochastic Algorithms for Minimax Optimization
under Polyak-Łojasiewicz Condition" (NeurIPS, 2022)

We consider the following two player Polyak-Łojasiewicz game 

$$
\begin{align*}
\min_{x\in\mathbb{R}^{d}}\max_{y\in\mathbb{R}^{d}} f(x,y) = \frac{1}{2} x^\top P x  - \frac{1}{2}y^\top Q  y + x^\top R  y,
\end{align*}
$$

where

$$
\begin{align*}
P = \frac{1}{n}\sum_{i=1}^n p_i p_i^\top, \quad
Q = \frac{1}{n}\sum_{i=1}^n q_i q_i^\top \quad \text{and} \quad
R = \frac{1}{n}\sum_{i=1}^n r_i r_i^\top.
\end{align*}
$$

To reimplement the experiments in our paper, please run 

```
code/PL_game/demo{i:%d}_by_{measurement:%s}.m
```
where $i$ denotes the experiment index, measurement can be 'dist' or 'gnorm', denoting that the convergence rate is measure by the distance to the unique saddle point $\Vert x - x^{\ast} \Vert^2 + \Vert y - y^{\ast} \Vert^2$ or the gradient norm $\Vert \nabla f(x,y) \Vert^2$.

We also provide a script to tune the parameters in the optimizers in

```
code/PL_game/tune.m
```

The datasets used in our experiments are available at https://www.csie.ntu.edu.tw/~cjlin/libsvm/
