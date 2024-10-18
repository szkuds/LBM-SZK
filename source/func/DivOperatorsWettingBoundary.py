import jax.numpy as jnp
from jax import Array

from source.SimulationSetup.SimulationParametersD2Q9 import r_0, r_1, h1, h2, kappa
from source.JaxGradLaplacian.JaxLaplacian2D import laplacian_2d
from source.JaxGradLaplacian.JaxGradient2D import grad_2d


def correction_density(rho):
    rho_ = rho[:, 0]
    rho_ = rho_.at[0:(r_0 + r_1)].set(rho[0:(r_0 + r_1), 0] + h1 / kappa)
    rho_ = rho_.at[(r_0 + r_1):-1].set(rho[(r_0 + r_1):-1, 0] + h2 / kappa)
    rho__ = jnp.column_stack((rho_, rho))
    return rho__


def laplacian(rho):
    rho__ = correction_density(rho)
    _rho = laplacian_2d(rho__)
    __rho = jnp.delete(_rho, 0, axis=1)
    return __rho


def grad_sq(rho):
    rho__ = correction_density(rho)
    _rho = grad_2d(rho__)
    __rho = jnp.delete(_rho, 0, axis=2)
    ___rho = jnp.stack((__rho[0] ** 2, __rho[0] * __rho[1], __rho[1] * __rho[0], __rho[1] ** 2), axis=0)
    return ___rho
