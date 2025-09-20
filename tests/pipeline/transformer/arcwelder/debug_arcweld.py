import matplotlib.pyplot as plt
import numpy as np
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand, ArcToCommand
from rayforge.pipeline.transformer.arcwelder import ArcWeld


def debug_arc_fitting():
    # Create test shape: ┌┐ with semicircle top
    ops = Ops()

    # Left vertical line
    ops.move_to(0, 0)
    ops.line_to(0, 5)

    # Semicircle (should become one arc_to)
    angles = np.linspace(np.pi / 2, -np.pi / 2, 20)
    for theta in angles:
        x = 5 + 5 * np.cos(theta)
        y = 5 + 5 * np.sin(theta)
        ops.line_to(x, y)

    # Right vertical line
    ops.line_to(10, 5)
    ops.line_to(10, 0)

    # Process with arcwelder
    original_commands = ops.commands.copy()
    welder = ArcWeld(tolerance=0.1, min_points=5)
    welder.run(ops)
    processed_commands = ops.commands

    # Plot visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original
    original_points = [
        cmd.end for cmd in original_commands if not cmd.is_state_command()
    ]
    x_orig, y_orig = zip(*original_points)
    ax1.plot(x_orig, y_orig, "b.-", label="Original")
    ax1.set_title("Original Path")
    ax1.axis("equal")

    # Plot processed
    x_proc, y_proc = [], []
    current_pos = None
    for cmd in processed_commands:
        match cmd:
            case MoveToCommand():
                current_pos = cmd.end
                x_proc.append(current_pos[0])
                y_proc.append(current_pos[1])
            case LineToCommand():
                x_proc.append(cmd.end[0])
                y_proc.append(cmd.end[1])
                current_pos = cmd.end
            case ArcToCommand():
                # Approximate arc with 50 points
                end_x, end_y = cmd.end
                i, j = cmd.center_offset
                radius = np.hypot(i, j)

                start_angle = np.arctan2(
                    current_pos[1] - j, current_pos[0] - i
                )
                end_angle = np.arctan2(end_y - j, end_x - i)

                angles = np.linspace(start_angle, end_angle, 50)
                if cmd.clockwise:
                    angles = angles[::-1]

                arc_points = [
                    (i + radius * np.cos(theta), j + radius * np.sin(theta))
                    for theta in angles
                ]
                x_proc.extend([p[0] for p in arc_points])
                y_proc.extend([p[1] for p in arc_points])
                current_pos = (end_x, end_y)

    ax2.plot(x_proc, y_proc, "r.-", label="Processed")
    ax2.set_title("ArcWelder Processed")
    ax2.axis("equal")

    plt.show()

    # Print command sequence for inspection
    print("\nOriginal commands:")
    for cmd in original_commands:
        print(f"{cmd}: {cmd.end}")

    print("\nProcessed commands:")
    for cmd in processed_commands:
        print(f"{cmd}: {cmd.args}")


debug_arc_fitting()
