import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def dim1():
    data = np.array([
        1., 2., 3., 4., 5., 6., 7., 8.,
    ])

    ar = pm.GaussianRandomWalk.dist(init_dist=pm.Normal.dist(0, 1),
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

def dim2():
    data = np.arange(14).reshape(7, 2)
    ar = pm.GaussianRandomWalk.dist(init_dist=pm.Normal.dist((0,1), 1),
                                    mu=2,sigma=1, steps=10)
    draws = pm.draw(ar, 3)
    print(draws)
    with pm.Model() as model:
        walk = pm.GaussianRandomWalk("walk", init_dist=pm.Normal.dist((0,1), 1),
                                    mu=2,sigma=1, steps=10)
        print(walk.type)
        obs = pm.Normal("obs", mu=walk[:,:7], sigma=1., observed=data.T)
        idata = pm.sample(draws=100,tune=100,chains=2,progressbar=True)
    az.plot_posterior(idata, var_names='walk')
    plt.show()
    return idata

def dim3():
    sigma = 0.1
    mu = np.arange(6).reshape(2, 3)
    init_dist = pm.Normal.dist(mu, sigma)
    steps = np.arange(2)*10
    rw = pm.GaussianRandomWalk.dist(
                               init_dist=init_dist,
                               mu=steps[:, np.newaxis],
                               sigma=sigma,
                               steps=5,
                               shape=(2, 3, 6))

    draws = pm.draw(rw, 1)
    print(draws)

if __name__ == '__main__':
    dim3()