import matplotlib.pyplot as plt
import numpy as np
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand, ArcToCommand
from rayforge.pipeline.transformer.arcwelder import ArcWeld


def visualize_segment_processing(segment, tolerance=0.1, min_points=3):
    """Visualize process_segment() behavior with command reconstruction"""
    welder = ArcWeld(tolerance=tolerance, min_points=min_points, max_points=30)
    ops = Ops()

    # Plot original segment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    x_orig = [p[0] for p in segment]
    y_orig = [p[1] for p in segment]
    ax1.plot(x_orig, y_orig, "bo-", label="Original Points")
    ax1.set_title("Original Segment")

    # Process the segment
    welder.process_segment(segment, ops)

    # Plot processed commands
    current_pos = None
    command_types = []
    arc_count = 0
    line_count = 0

    for cmd in ops.commands:
        match cmd:
            case MoveToCommand():
                current_pos = cmd.end
                ax2.plot(
                    current_pos[0],
                    current_pos[1],
                    "go",
                    markersize=8,
                    label="MoveTo",
                )
                command_types.append(("move_to", current_pos))
            case LineToCommand():
                end = cmd.end
                ax2.plot(
                    [current_pos[0], end[0]],
                    [current_pos[1], end[1]],
                    "r--",
                    linewidth=2,
                    label="LineTo" if line_count == 0 else "",
                )
                current_pos = end
                line_count += 1
                command_types.append(("line_to", end))
            case ArcToCommand():
                end_x, end_y = cmd.end
                i, j = cmd.center_offset
                center = (current_pos[0] + i, current_pos[1] + j)
                radius = np.hypot(i, j)

                # Calculate angles
                start_angle = np.arctan2(
                    current_pos[1] - center[1], current_pos[0] - center[0]
                )
                end_angle = np.arctan2(end_y - center[1], end_x - center[0])

                # Generate arc points
                theta = np.linspace(start_angle, end_angle, 50)
                if cmd.clockwise:
                    theta = theta[::-1]

                arc_points = [
                    (
                        center[0] + radius * np.cos(t),
                        center[1] + radius * np.sin(t),
                    )
                    for t in theta
                ]

                ax2.plot(
                    [p[0] for p in arc_points],
                    [p[1] for p in arc_points],
                    "m-",
                    linewidth=2,
                    label="ArcTo" if arc_count == 0 else "",
                )
                ax2.plot(
                    center[0],
                    center[1],
                    "cx",
                    markersize=10,
                    label="Arc Center",
                )
                current_pos = end_x, end_y
                arc_count += 1
                command_types.append(("arc_to", (end_x, end_y)))

    ax2.set_title(
        f"Processed Commands\n(Tolerance={tolerance}, MinPoints={min_points})"
    )

    # Add annotations
    for i, (cmd_type, pos) in enumerate(command_types):
        ax2.annotate(
            f"{i}: {cmd_type}",
            pos,
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="darkblue",
        )

    # Configure both plots
    for ax in (ax1, ax2):
        ax.grid(True)
        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


# Test Case 1: Perfect semicircle + straight line
segment = [
    (10 * np.cos(theta), 10 * np.sin(theta))
    for theta in np.linspace(0, np.pi, 20)
] + [(x, 0) for x in np.linspace(-10, -15, 5)]

"""
# Test Case 2: Mixed valid/invalid arcs
segment = [
    (5*np.cos(theta), 5*np.sin(theta))
    for theta in np.linspace(0, 0.7*np.pi, 5)
] + [(x, x*0.1) for x in np.linspace(5, 10, 5)]

# Test Case 3: Near-colinear with small arc
segment = [
    (x, 0.1*x) for x in np.linspace(0, 5, 10)
] + [
    (5 + 2*np.cos(theta), 0.5 + 2*np.sin(theta))
    for theta in np.linspace(0, np.pi, 10)
]
"""

# Run visualization
visualize_segment_processing(segment, tolerance=0.1, min_points=5)
