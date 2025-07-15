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
