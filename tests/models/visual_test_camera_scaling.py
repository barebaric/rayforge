import cv2
import numpy as np
from rayforge.models.camera import Camera

def run_visual_test():
    # Create a mock camera instance
    camera = Camera("Visual Test Camera", "0")

    # Create a dummy image with a grid pattern for visual verification
    raw_image = np.zeros((400, 400, 3), dtype=np.uint8)  # Black background

    # Draw a grid
    for i in range(0, 400, 20):
        cv2.line(raw_image, (i, 0), (i, 399), (0, 255, 0), 1)  # Green vertical lines
        cv2.line(raw_image, (0, i), (399, i), (0, 255, 0), 1)  # Green horizontal lines

    # Draw a red square at a known location
    cv2.rectangle(raw_image, (100, 100), (150, 150), (0, 0, 255), -1) # Red square
    cv2.rectangle(raw_image, (200, 200), (250, 250), (255, 0, 0), -1) # Blue square
    cv2.rectangle(raw_image, (300, 300), (350, 350), (0, 255, 255), -1) # Yellow square

    camera._image_data = raw_image

    # Define corresponding points to introduce some perspective distortion
    image_points = [(50, 50), (350, 50), (380, 350), (20, 350)]
    world_points = [(0, 0), (200, 0), (200, 200), (0, 200)]
    camera.image_to_world = image_points, world_points

    output_size = (400, 400)  # Desired output image size in pixels
    physical_area = ((0, 0), (200, 200))  # The 200x200mm area we want to view

    # Display raw image with image_points marked
    raw_image_display = raw_image.copy()
    for i, point in enumerate(image_points):
        cv2.circle(raw_image_display, tuple(map(int, point)), 5, (0, 255, 255), -1) # Yellow circles
        cv2.putText(raw_image_display, f"P{i}", (int(point[0]) + 10, int(point[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Raw Image with Image Points", raw_image_display)
    print("Displaying 'Raw Image with Image Points'. Close window to continue.")
    # Wait for a key press or window close for the first window
    while cv2.getWindowProperty("Raw Image with Image Points", cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break
    cv2.destroyWindow("Raw Image with Image Points")

    # Test the general get_work_surface_image functionality
    aligned_image = camera.get_work_surface_image(output_size, physical_area)

    if aligned_image is not None:
        if aligned_image.shape[2] == 3:
            display_image = aligned_image
        elif aligned_image.shape[2] == 4:
            display_image = cv2.cvtColor(aligned_image, cv2.COLOR_RGBA2BGR)
        else:
            display_image = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)

        # Draw expected physical area boundaries on the output image
        cv2.rectangle(display_image, (0, 0), (399, 399), (0, 0, 255), 2) # Red border

        # Mark expected positions of world_points on the transformed image
        # Convert world_points (mm) to pixel coordinates on the output image
        # 1mm = 2 pixels (since 200mm maps to 400px)
        world_points_pixels = [
            (int(p[0] * 2), int(p[1] * 2)) for p in world_points
        ]

        for i, point_px in enumerate(world_points_pixels):
            cv2.circle(display_image, point_px, 5, (255, 0, 255), -1) # Magenta circles
            cv2.putText(display_image, f"W{i}", (point_px[0] + 10, point_px[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Add text labels for clarity
        cv2.putText(display_image, "Expected 0,0mm (Top-Left)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_image, "Expected 200,0mm (Top-Right)", (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_image, "Expected 0,200mm (Bottom-Left)", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_image, "Expected 200,200mm (Bottom-Right)", (250, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Transformed Work Surface Image", display_image)
        print("Displaying 'Transformed Work Surface Image'. Close window to continue.")
    else:
        print("Failed to generate work surface image.")

    # Wait for a key press and close windows
    # Wait for a key press or window close for the second window
    while cv2.getWindowProperty("Transformed Work Surface Image", cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break
    cv2.destroyWindow("Transformed Work Surface Image")

if __name__ == "__main__":
    print("Running visual test for get_work_surface_image.")
    print("First, a 'Raw Image with Image Points' window will appear.")
    print("  - This shows the original image with yellow circles marking the 'image_points'.")
    print("  - These points define the skewed region in the raw image that will be transformed.")
    print("Close this window to proceed.")
    print("\nNext, a 'Transformed Work Surface Image' window will appear.")
    print("  - This is the result of get_work_surface_image, showing the 'un-skewed' view.")
    print("  - A red border outlines the expected 200x200mm physical area (400x400 pixels).")
    print("  - Magenta circles mark the 'world_points' (0,0), (200,0), (200,200), (0,200)mm,")
    print("    which should align with the corners of the red border.")
    print("  - The green grid lines should appear straight and evenly spaced.")
    print("  - The colored squares should be visible and their shapes corrected, appearing")
    print("    at their expected relative positions within the un-skewed grid.")
    print("Close this window to finish the test.")
    run_visual_test()
