import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow
from numpy.typing import NDArray


def draw_policy(image: NDArray, arrows: NDArray, starting_point: NDArray):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Calculate the number of arrows
    num_arrows = len(arrows)

    # Create start positions array
    start_positions = np.tile(starting_point, (num_arrows, 1))

    # Draw the arrows
    for start, arrow_vector in zip(start_positions, arrows):
        arrow = Arrow(
            start[0],
            start[1],
            arrow_vector[0],
            arrow_vector[1],
            width=0.5,
            color="r",
        )
        ax.add_patch(arrow)

    # Set the axis limits to match the image size
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Reverse y-axis to match image coordinates

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

    return fig, ax
