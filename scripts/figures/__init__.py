"""Generators for the website documentation figures.

Each module in this package renders one figure and returns its images
from ``generate()`` as ``(filename, image)`` pairs. Regenerate them
with::

    pixi run figures            # all figures
    pixi run figures bidir_offset path_optimization

The images are written to ``website/static/images/`` as lossless
webp. They carry no translatable words; titles live in the markdown
that embeds them.
"""
