import jax.numpy as jnp

"""
Code by Sacha Szkudlarek
"""

"""
Args:
    grid (jnp.ndarray): The input 2D grid.
    
Returns:
    jnp.ndarray: The Laplacian of the input grid.
"""


def laplacian_2d(grid):
    laplacian = jnp.zeros_like(grid)  # Initialize a new array with the same shape as the input grid, filled with zeros

    # bulk
    bulk = laplacian.at[1:-2, 1:-2].set(  # Compute the Laplacian for the bulk (interior) nodes
        grid[2:-1, 1:-2] +  # Contribution from nodes to the right
        grid[0:-3, 1:-2] +  # Contribution from nodes to the left
        grid[1:-2, 2:-1] +  # Contribution from nodes below
        grid[1:-2, 0:-3] -  # Contribution from nodes above
        4 * grid[1:-2, 1:-2]  # Subtract 4 times the central node value
    )

    # left
    left = laplacian.at[0, 1:-2].set(  # Compute the Laplacian for the left edge nodes
        grid[1, 1:-2] +  # Contribution from nodes to the right
        grid[0, 0:-3] +  # Contribution from nodes below
        grid[0, 2:-1] -  # Contribution from nodes above
        3 * grid[0, 1:-2]  # Subtract 3 times the central node value
    )

    # right
    right = laplacian.at[-1, 1:-2].set(  # Compute the Laplacian for the bottom edge nodes
        grid[-2, 1:-2] +  # Contribution from nodes to the left
        grid[-1, 0:-3] +  # Contribution from nodes below
        grid[-1, 2:-1] -  # Contribution from nodes above
        3 * grid[-1, 1:-2]  # Subtract 3 times the central node value
    )

    # bottom
    bottom = laplacian.at[1:-2, 0].set(  # Compute the Laplacian for the left edge nodes
        grid[1:-2, 1] +  # Contribution from nodes above
        grid[0:-3, 0] +  # Contribution from nodes to the left
        grid[2:-1, 0] -  # Contribution from nodes to the right 
        3 * grid[1:-2, 0]  # Subtract 3 times the central node value
    )

    # top
    top = laplacian.at[1:-2, -1].set(  # Compute the Laplacian for the right edge nodes
        grid[1:-2, -2] +  # Contribution from nodes below
        grid[0:-3, -1] +  # Contribution from nodes to the left
        grid[2:-1, -1] -  # Contribution from nodes to the right
        3 * grid[1:-2, -1]  # Subtract 3 times the central node value
    )

    # bottom-left
    bottomleft = laplacian.at[0, 0].set(  # Compute the Laplacian for the top-left corner node
        grid[0, 1] +  # Contribution from node above
        grid[1, 0] -  # Contribution from node to the right
        2 * grid[0, 0]  # Subtract 2 times the central node value
    )

    # topleft
    topleft = laplacian.at[0, -1].set(  # Compute the Laplacian for the top-right corner node
        grid[0, -2] +  # Contribution from node below
        grid[1, -1] -  # Contribution from node to the right
        2 * grid[0, -1]  # Subtract 2 times the central node value
    )

    # bottomright
    bottomright = laplacian.at[-1, 0].set(  # Compute the Laplacian for the bottom-left corner node
        grid[-2, 0] +  # Contribution from node to the left
        grid[-1, 1] -  # Contribution from node above
        2 * grid[-1, 0]  # Subtract 2 times the central node value
    )

    # topright
    topright = laplacian.at[-1, -1].set(  # Compute the Laplacian for the bottom-right corner node
        grid[-1, -2] +  # Contribution from node below
        grid[-2, -1] -  # Contribution from node to the left
        2 * grid[-1, -1]  # Subtract 2 times the central node value
    )

    return bulk + top + bottom + left + right + topleft + topright + bottomleft + bottomright
