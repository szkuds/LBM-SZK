import jax.numpy as jnp

"""
Code by Sacha Szkudlarek
"""
"""
Computes the first order gradient of a 2D grid using finite differences.

Args:
    grid (jnp.ndarray): The input 2D grid.

Returns: jnp.ndarray: A 3D array where the first dimension represents the x and y components of the gradient, 
and the other two dimensions match the input grid shape."""


def grad_2d(grid):
    # Initialize a 3D array to store the x and y components of the gradient
    grad = jnp.zeros((2, grid.shape[0], grid.shape[1]))

    # Compute the x gradient for the bulk (interior) points
    bulk_x = grad.at[0, 1:-2, :].set((
                                             grid[0:-3, :] -  # Contribution from node to the right
                                             grid[2:-1, :]  # Contribution from node to the left
                                     ) / 2
                                     )

    # Compute the y gradient for the bulk (interior) points  
    bulk_y = grad.at[1, :, 1:-2].set((
                                             grid[:, 2:-1] -  # Contribution from node above
                                             grid[:, 0:-3]  # Contribution from node below
                                     ) / 2
                                     )

    # Compute the x gradient for the left edge
    left_x = grad.at[0, 0, :].set((
            grid[0, :] -  # Contribution from the edge
            grid[1, :]  # Contribution from the right
    )
    )

    # Compute the x gradient for the right edge
    right_x = grad.at[0, -1, :].set((
            grid[-2, :] -  # Contribution from the left
            grid[-1, :]  # Contribution from the edge
    )
    )

    # Compute the y gradient for the bottom edge
    bottom_y = grad.at[1, :, 0].set((
            grid[:, 0] -  # Contribution from the edge
            grid[:, 1]  # Contribution from the right
    )
    )

    # Compute the y gradient for the top edge
    top_y = grad.at[1, :, -1].set((
            grid[:, -2] -  # Contribution from the bottom
            grid[:, -1]  # Contribution from the edge
    )
    )

    # Combine the contributions from bulk and edges to get the final gradient
    return bulk_x + bulk_y + left_x + right_x + bottom_y + top_y
