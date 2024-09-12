from source.SimulationSetup.SimulationParametersD2Q9 import *


def init():
    x = jnp.arange(nx) + 0.5
    y = jnp.arange(ny) + 0.5
    [x, y] = jnp.meshgrid(x, y)
    u = jnp.zeros((2, nx, ny))
    rho = rho0 * jnp.ones((nx, ny))
    r0 = r_0 / 1.5 / (jnp.sin(math.pi - theta_3) * 2)
    center_y, center_x = 1 / jnp.tan(math.pi - theta_3) * r_0 / 1.5 * 0.5, r_0
    distance = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = distance <= r0
    rho = rho.at[mask.T].set(rho_c * (1 + jnp.sqrt(beta_tau)))
    return [rho, u]


def init_():
    u = jnp.zeros((2, nx, ny))
    rho = rho0 * jnp.ones((nx, ny))
    rho = rho.at[int(nx / 4):int(3 * nx / 4), int(ny / 4):int(ny * 3 / 4)].set(rho_c * (1 + jnp.sqrt(beta_tau)))
    return [rho, u]
