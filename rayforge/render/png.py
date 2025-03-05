import pyvips
from .vips import VipsRenderer


class PNGRenderer(VipsRenderer):
    label = 'PNG files'
    mime_types = ('image/png',)
    extensions = ('.png',)

    @classmethod
    def get_vips_loader(cls):
        return pyvips.Image.pngload_buffer

    @classmethod
    def get_vips_loader_args(cls):
        return {"access": pyvips.Access.RANDOM}  # Stream-friendly mode

    @classmethod
    def get_vips_image(cls,
                       data,
                       width_px=None,
                       height_px=None,
                       pixels_per_mm=None):
        try:
            image = cls.get_vips_loader()(data, **cls.get_vips_loader_args())
        except pyvips.error.Error as e:
            raise ValueError(f"Failed to load PNG: {e}")

        # 1. Calculate target dimensions
        target_width, target_height = cls._calculate_target_size(
            image, width_px, height_px, pixels_per_mm
        )

        # 2. Apply scaling with pyramid reduction for large images
        if image.width > target_width or image.height > target_height:
            image = image.thumbnail_image(
                min(target_width, target_height),
                height=target_height,
                size=pyvips.Size.DOWN  # Prevent upscaling
            )

        # 3. Add alpha channel if needed
        if image.bands == 3:
            image = image.bandjoin(255)

        return image

    @classmethod
    def _calculate_target_size(cls, image, width_px, height_px, pixels_per_mm):
        """Calculate safe target dimensions with multiple fallbacks"""
        # Priority 1: Explicit pixel dimensions
        if width_px or height_px:
            return width_px or image.width, height_px or image.height

        # Priority 2: Pixels-per-mm calculation
        if pixels_per_mm and None not in pixels_per_mm:
            nat_width_mm, nat_height_mm = cls.get_natural_size(image)
            if nat_width_mm and nat_height_mm:
                return (
                    int(nat_width_mm * pixels_per_mm[0]),
                    int(nat_height_mm * pixels_per_mm[1])
                )

        # Fallback: Original dimensions
        return image.width, image.height

    @classmethod
    def get_natural_size(cls, data, px_factor=0):
        image = cls.get_vips_loader()(data, **cls.get_vips_loader_args())

        # Get resolution with safe defaults
        try:
            xres = image.get('xres')  # pixels per mm
        except pyvips.error.Error:
            xres = 5.0

        try:
            yres = image.get('yres')
        except pyvips.error.Error:
            yres = 5.0

        # Convert DPI to mm dimensions
        mm_width = image.width / xres if xres > 0 else None
        mm_height = image.height / yres if yres > 0 else None

        return mm_width, mm_height

    @classmethod
    def prepare(cls, data):
        # Process in streaming mode to avoid full decode
        image = pyvips.Image.new_from_buffer(
            data,
            "",
            access=pyvips.Access.SEQUENTIAL
        )

        # Simple passthrough with alpha check
        if image.bands == 3:
            image = image.bandjoin(255)

        return image.write_to_buffer('.png', strip=True, compression=6)
