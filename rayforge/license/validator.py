import logging
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Callable, Dict, Optional

from blinker import Signal

from .gumroad_provider import GumroadProvider
from .patreon_provider import PatreonProvider
from .provider import (
    LicenseProvider,
    LicenseResult,
    LicenseStatus,
)


logger = logging.getLogger(__name__)


class LicenseValidator:
    CACHE_DURATION = timedelta(hours=24)

    def __init__(
        self, config_dir: Path, patreon_client_id: Optional[str] = None
    ):
        self._gumroad: GumroadProvider = GumroadProvider(config_dir)
        self._patreon: Optional[PatreonProvider] = None
        self._cache: Dict[str, tuple] = {}
        self.changed = Signal()

        if patreon_client_id:
            self._patreon = PatreonProvider(config_dir, patreon_client_id)

    @property
    def providers(self) -> Dict[str, LicenseProvider]:
        result: Dict[str, LicenseProvider] = {"gumroad": self._gumroad}
        if self._patreon:
            result["patreon"] = self._patreon
        return result

    def validate(self, addon_id: str, license_config: Dict) -> LicenseResult:
        cached = self._cache.get(addon_id)
        if cached:
            result, timestamp = cached
            if datetime.now() - timestamp < self.CACHE_DURATION:
                return result

        has_gumroad = license_config.get("product_ids") or license_config.get(
            "product_id"
        )
        has_patreon = license_config.get("patreon_tier_ids")

        if has_gumroad:
            if self._gumroad.is_configured():
                result = self._gumroad.validate(license_config)
                if result.status == LicenseStatus.VALID:
                    self._cache[addon_id] = (result, datetime.now())
                    return result

        if has_patreon and self._patreon:
            if self._patreon.is_configured():
                result = self._patreon.validate(license_config)
                if result.status == LicenseStatus.VALID:
                    self._cache[addon_id] = (result, datetime.now())
                    return result

        if has_gumroad and has_patreon:
            message = (
                "License required. Purchase on Gumroad or link Patreon "
                "account."
            )
        elif has_gumroad:
            message = "License required. Purchase on Gumroad."
        elif has_patreon:
            message = "License required. Link your Patreon account."
        else:
            message = "License required."

        result = LicenseResult(status=LicenseStatus.NOT_FOUND, message=message)
        self._cache[addon_id] = (result, datetime.now())
        return result

    def check_license(
        self, addon_id: str, license_config: Dict
    ) -> tuple[bool, str, str]:
        """
        Check if addon requires and has valid license.

        Returns:
            Tuple of (is_allowed, message, purchase_url)
        """
        if not license_config or not license_config.get("required"):
            return True, "", ""

        result = self.validate(addon_id, license_config)
        purchase_url = license_config.get("purchase_url", "")

        if result.status == LicenseStatus.VALID:
            return True, result.message, ""

        return False, result.message, purchase_url

    def invalidate_cache(self, addon_id: Optional[str] = None) -> None:
        if addon_id:
            self._cache.pop(addon_id, None)
        else:
            self._cache.clear()

    def get_provider(self, name: str) -> Optional[LicenseProvider]:
        if name == "gumroad":
            return self._gumroad
        if name == "patreon":
            return self._patreon
        return None

    def has_gumroad_license_for(self, product_id: str) -> bool:
        return self._gumroad.has_license_for(product_id)

    def add_gumroad_license(self, product_id: str, license_key: str) -> None:
        self._gumroad.add_license(product_id, license_key)
        self.invalidate_cache()
        self.changed.send(self)

    def remove_gumroad_license(self, product_id: str) -> None:
        self._gumroad.remove_license(product_id)
        self.invalidate_cache()
        self.changed.send(self)

    def get_gumroad_licenses(self) -> Dict[str, str]:
        return self._gumroad.get_licenses()

    def is_patreon_linked(self) -> bool:
        return self._patreon is not None and self._patreon.is_configured()

    def unlink_patreon(self) -> None:
        if self._patreon:
            self._patreon.unlink()
            self.invalidate_cache()

    def start_patreon_oauth(
        self, on_complete: Callable[[bool, Optional[str]], None]
    ) -> Optional[tuple[int, Thread]]:
        if not self._patreon:
            return None
        return self._patreon.start_oauth_flow(on_complete)

    def get_patreon_oauth_url(self) -> Optional[str]:
        if not self._patreon:
            return None
        return self._patreon.get_oauth_url()
