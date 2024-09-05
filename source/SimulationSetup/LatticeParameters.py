import jax.numpy as jnp

# Lattice parameters
#    Grid index Numbers
#    6 2 5
#     \|/
#    3-0-1
#     /|\
#    7 4 8

n_discrete_velocities = 9

lattice_velocities = jnp.array([[0, 1, 0, -1, 0, 1, 1, -1, -1],  # Velocities x components
                                [0, 0, 1, 0, -1, 1, -1, -1, 1]])  # Velocities y components
lattice_index = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
opposite_lattice_indices = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
velocities_right = jnp.array([1, 5, 8])
velocities_left = jnp.array([3, 6, 7])
velocities_down = jnp.array([4, 7, 8])
velocities_up = jnp.array([2, 5, 6])
velocities_ver = jnp.array([0, 2, 4])
velocities_hor = jnp.array([0, 1, 3])

w_i = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # Weights_i, Note that the 0
# node is first, LBM_book 9.2.1.3
w_xx = jnp.array([0, 1 / 3, -1 / 6, 1 / 3, -1 / 6, -1 / 24, -1 / 24, -1 / 24, -1 / 24])  # Weights_xx
w_yy = jnp.array([0, -1 / 6, 1 / 3, -1 / 6, 1 / 3, -1 / 24, -1 / 24, -1 / 24, -1 / 24])  # Weights_yy
w_xy = jnp.array([0, 0, 0, 0, 0, 1 / 4, -1 / 4, 1 / 4, -1 / 4])  # Weights_xy

w_i_2 = jnp.stack((w_xx, w_xy, w_xy, w_yy), axis=0)