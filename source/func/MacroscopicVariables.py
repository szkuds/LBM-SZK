import jax.numpy as jnp


def get_density(f_i):
    return jnp.einsum('xyi->xy', f_i)


def get_velocity(f_i, force, c_i):
    rho = jnp.einsum('xyi->xy', f_i)
    u = jnp.einsum('ai,xyi->axy', c_i, f_i) + 0.5 * force
    return u / rho
