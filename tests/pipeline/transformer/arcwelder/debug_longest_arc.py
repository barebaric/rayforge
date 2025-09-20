import matplotlib.pyplot as plt
import numpy as np
from rayforge.pipeline.transformer.arcwelder import ArcWeld


def visualize_arc_selection(points, tolerance=0.1, min_points=3):
    """Visualize arc finding process with candidate segments"""
    welder = ArcWeld(tolerance=tolerance, min_points=min_points)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot original points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    ax.plot(x, y, "bo-", label="Original Path")

    # Simulate arc finding process
    start_index = 0
    n = len(points)
    processed = []

    while start_index < n:
        best_arc, best_end = welder._find_longest_valid_arc(
            points, start_index
        )

        if best_arc and (best_end - start_index) >= min_points:
            # Plot candidate arc segment
            seg = points[start_index:best_end]
            ax.plot(
                [p[0] for p in seg],
                [p[1] for p in seg],
                "rx--",
                linewidth=2,
                markersize=8,
                label=f"Arc Candidate {len(processed)}",
            )

            # Plot fitted circle
            center, radius, _ = best_arc
            theta = np.linspace(0, 2 * np.pi, 100)
            xc = center[0] + radius * np.cos(theta)
            yc = center[1] + radius * np.sin(theta)
            ax.plot(xc, yc, ":", color="orange", alpha=0.5)

            processed.append((start_index, best_end))
            start_index = best_end
        else:
            start_index += 1

    # Annotate results
    for i, (s, e) in enumerate(processed):
        ax.annotate(
            f"Arc {i}\n({s}-{e - 1})",
            (points[s][0], points[s][1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_title(
        f"Arc Selection Process (Tolerance={tolerance}, "
        f"Min Points={min_points})"
    )
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")
    plt.show()


# Test Case 1: Valid arc followed by straight line
points = [
    (5 * np.cos(theta), 5 * np.sin(theta))
    for theta in np.linspace(0, np.pi, 20)
] + [(x, 0) for x in np.linspace(-5, -10, 5)]

# Test Case 2: Two consecutive arcs
points = [
    (5 * np.cos(theta), 5 * np.sin(theta))
    for theta in np.linspace(0, np.pi, 15)
] + [
    (10 + 3 * np.cos(theta), 5 + 3 * np.sin(theta))
    for theta in np.linspace(np.pi, 1.5 * np.pi, 15)
]

# Test Case 3: Noisy arc with valid/invalid sections
points = [
    (
        5 * np.cos(theta) + np.random.normal(0, 0.1),
        5 * np.sin(theta) + np.random.normal(0, 0.1),
    )
    for theta in np.linspace(0, 0.8 * np.pi, 25)
]

# Run visualization with different parameters
visualize_arc_selection(points, tolerance=0.15, min_points=5)
