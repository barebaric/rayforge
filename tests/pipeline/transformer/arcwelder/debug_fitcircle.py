import matplotlib.pyplot as plt
import numpy as np
from rayforge.core.geo.analysis import fit_circle_to_points


def visualize_circle_fit(points, title):
    """Plot original points vs fitted circle with error visualization"""
    result = fit_circle_to_points(points)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot original points
    x_orig = [p[0] for p in points]
    y_orig = [p[1] for p in points]
    ax.plot(x_orig, y_orig, "bo", label="Original Points")

    if result:
        (xc, yc), r, error = result

        # Plot fitted circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x_fit = xc + r * np.cos(theta)
        y_fit = yc + r * np.sin(theta)
        ax.plot(x_fit, y_fit, "r--", label="Fitted Circle")
        ax.plot(xc, yc, "rx", markersize=10, label="Fitted Center")

        # Plot error indicators
        for x, y in points:
            dx = x - xc
            dy = y - yc
            dist = np.hypot(dx, dy)
            ax.plot(
                [x, xc + dx * r / dist],
                [y, yc + dy * r / dist],
                "g:",
                alpha=0.5,
            )

        # Annotation box
        textstr = "\n".join(
            (
                f"Center: ({xc:.3f}, {yc:.3f})",
                f"Radius: {r:.3f}",
                f"Max Error: {error:.3f}",
            )
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.9)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    plt.show()


# Test case 1: Perfect semicircle (should match perfectly)
def test_semicircle():
    angles = np.linspace(0, np.pi, 20)
    return [(5 + 5 * np.cos(theta), 5 * np.sin(theta)) for theta in angles]


# Test case 2: Noisy quarter-circle (should show small error)
def test_noisy_quartercircle():
    np.random.seed(42)
    angles = np.linspace(np.pi / 2, np.pi, 15)
    return [
        (
            3 + 2 * np.cos(theta) + np.random.normal(0, 0.05),
            4 + 2 * np.sin(theta) + np.random.normal(0, 0.05),
        )
        for theta in angles
    ]


# Test case 3: Near-colinear points (should fail circle fitting)
def test_near_colinear():
    return [(0, 0), (2, 0.1), (4, 0.2), (6, 0.3), (8, 0.4)]


# Run visualizations
visualize_circle_fit(test_semicircle(), "Perfect Semicircle")
visualize_circle_fit(test_noisy_quartercircle(), "Noisy Quarter-circle")
visualize_circle_fit(test_near_colinear(), "Near-colinear Points")
