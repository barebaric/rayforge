import matplotlib.pyplot as plt
import numpy as np
from rayforge.core.ops import Ops, MoveToCommand, ArcToCommand
from rayforge.pipeline.transformer.arcwelder import ArcWeld
from rayforge.core.geo.analysis import fit_circle_to_points


def debug_semicircle():
    """Visualize semicircle fitting and arc generation"""
    radius = 10.0
    points = [
        (radius * np.cos(theta), radius * np.sin(theta))
        for theta in np.linspace(0, np.pi, 20)
    ]

    # Fit circle
    result = fit_circle_to_points(points)
    print("Fit Result:", result)

    # Process segment
    welder = ArcWeld(tolerance=0.1, max_points=20)
    ops = Ops()
    welder.process_segment(points, ops)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Original points
    ax1.plot(*zip(*points), "bo-")
    ax1.set_title("Original Points")

    # Processed commands
    current_pos = None
    for cmd in ops:
        match cmd:
            case MoveToCommand():
                current_pos = cmd.end
                ax2.plot(*current_pos, "go")
            case ArcToCommand():
                end_x, end_y = cmd.end
                i, j = cmd.center_offset
                center = (current_pos[0] + i, current_pos[1] + j)
                theta = np.linspace(0, np.pi, 50)
                x = center[0] + np.hypot(i, j) * np.cos(theta)
                y = center[1] + np.hypot(i, j) * np.sin(theta)
                ax2.plot(x, y, "r--")
                current_pos = (end_x, end_y)

    ax2.set_title("Processed Arc")
    plt.show()


# Run debug
debug_semicircle()
