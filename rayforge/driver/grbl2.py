import aiohttp
from ..debug import debug_log_manager, LogType
from .grbl import GrblDriver, upload_url


class Grbl2Driver(GrblDriver):
    """
    A next-generation driver for GRBL-compatible controllers that use
    a modern, standard file upload API. Inherits from the base GrblDriver
    for all other functionality.
    """

    # These labels will appear in the UI's driver selection list.
    label = _("GRBL (experimental)")
    subtitle = _("Next generation GRBL-via-network driver")

    async def _upload(self, gcode, filename):
        """
        Overrides the base GrblDriver's upload method with a standard
        multipart/form-data POST request.
        """
        form = aiohttp.FormData()

        # Use a standard field name for the file
        form.add_field(
            "file", gcode, filename=filename, content_type="text/plain"
        )

        # The upload path is specified as a query parameter in the URL.
        url = f"{self.http_base}{upload_url}?path=/"

        debug_log_manager.add_entry(
            self.__class__.__name__,
            LogType.TX,
            f"POST to {url} with file '{filename}' size {len(gcode)}",
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
                # Raise an exception for bad status codes (4xx or 5xx)
                response.raise_for_status()
                data = await response.text()

        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data.encode("utf-8")
        )
        return data
