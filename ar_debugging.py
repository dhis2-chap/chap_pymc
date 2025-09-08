import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
data = np.array([
    1., 2., 3., 4., 5., 6., 7., 8.,
])

ar = pm.GaussianRandomWalk.dist(init_dist=pm.Normal.dist(0,1 ),
                                mu=0,
                                sigma=1,
                                steps=10)
draws = pm.draw(ar, 3)
with pm.Model() as model:
    walk = pm.GaussianRandomWalk(
        "walk", init_dist=pm.Normal.dist(0, 1), sigma=1, steps=10
    )
    obs = pm.Normal('obs', mu=walk[:8], sigma=1., observed=data)
    idata = pm.sample(draws=100,
                    tune=100,
                    chains=2,
                    progressbar=True)

az.plot_posterior(idata, var_names='walk')
plt.show()