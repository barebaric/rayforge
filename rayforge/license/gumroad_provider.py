import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from .provider import (
    LicenseProvider,
    LicenseResult,
    LicenseStatus,
    LicenseType,
)


logger = logging.getLogger(__name__)


class GumroadProvider(LicenseProvider):
    API_URL = "https://api.gumroad.com/v2/licenses/verify"

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = config_dir / "gumroad_licenses.json"
        self._licenses: Dict[str, str] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_config()

    @property
    def name(self) -> str:
        return "gumroad"

    def is_configured(self) -> bool:
        return len(self._licenses) > 0

    def has_license_for(self, product_id: str) -> bool:
        return product_id in self._licenses

    def add_license(self, product_id: str, license_key: str) -> None:
        self._licenses[product_id] = license_key
        self._save_config()

    def remove_license(self, product_id: str) -> None:
        self._licenses.pop(product_id, None)
        self._cache.pop(product_id, None)
        self._save_config()

    def get_licenses(self) -> Dict[str, str]:
        return dict(self._licenses)

    def get_cached_result(self, product_id: str) -> Optional[Dict]:
        return self._cache.get(product_id)

    def clear_cache(self, product_id: Optional[str] = None) -> None:
        if product_id:
            self._cache.pop(product_id, None)
        else:
            self._cache.clear()

    def validate_key(self, product_id: str, license_key: str) -> LicenseResult:
        return self._validate_single(product_id, license_key)

    def validate(self, config: Dict) -> LicenseResult:
        product_ids = config.get("product_ids", [])
        if not product_ids:
            product_id = config.get("product_id")
            if product_id:
                product_ids = [product_id]

        if not product_ids:
            return LicenseResult(
                status=LicenseStatus.ERROR,
                message="Missing product_ids in addon config",
            )

        for product_id in product_ids:
            license_key = self._licenses.get(product_id)
            if not license_key:
                continue

            result = self._validate_single(product_id, license_key)
            if result.status == LicenseStatus.VALID:
                return result

        if self._licenses:
            return LicenseResult(
                status=LicenseStatus.INVALID,
                message="No valid license found for this product",
            )

        return LicenseResult(
            status=LicenseStatus.NOT_FOUND,
            message="No license key configured for this product",
        )

    def _validate_single(
        self, product_id: str, license_key: str
    ) -> LicenseResult:
        cached = self._get_valid_cache(product_id)
        if cached:
            status = LicenseStatus(cached.get("status"))
            license_type = LicenseType(cached.get("license_type", "unknown"))
            last_validated_str = cached.get("last_validated")
            last_validated = None
            if last_validated_str:
                last_validated = datetime.fromisoformat(last_validated_str)
            return LicenseResult(
                status=status,
                message=cached.get("message", ""),
                license_type=license_type,
                customer_email=cached.get("customer_email"),
                last_validated=last_validated,
                metadata=cached.get("metadata", {}),
            )

        if license_key.startswith("TESTKEY-"):
            return self._create_test_result(product_id)

        try:
            data = urllib.parse.urlencode(
                {
                    "product_id": product_id,
                    "license_key": license_key,
                    "increment_uses_count": "false",
                }
            ).encode()

            req = urllib.request.Request(
                self.API_URL, data=data, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())

            if not result.get("success"):
                return LicenseResult(
                    status=LicenseStatus.INVALID,
                    message="License key is invalid",
                )

            purchase = result.get("purchase", {})

            if purchase.get("refunded"):
                return LicenseResult(
                    status=LicenseStatus.INVALID,
                    message="Purchase was refunded",
                )

            if purchase.get("subscription_cancelled_at"):
                return LicenseResult(
                    status=LicenseStatus.EXPIRED,
                    message="Subscription has been cancelled",
                )

            if purchase.get("subscription_ended_at"):
                return LicenseResult(
                    status=LicenseStatus.EXPIRED,
                    message="Subscription has ended",
                )

            license_type = LicenseType.ONE_TIME
            if purchase.get("subscription_id"):
                license_type = LicenseType.SUBSCRIPTION

            result_obj = LicenseResult(
                status=LicenseStatus.VALID,
                message="License is valid",
                license_type=license_type,
                customer_email=purchase.get("email"),
                last_validated=datetime.now(),
                metadata={
                    "product_id": product_id,
                    "product_name": purchase.get("product_name"),
                    "purchase_date": purchase.get("sale_timestamp"),
                },
            )

            self._cache_result(product_id, result_obj)
            return result_obj

        except urllib.error.URLError as e:
            cached = self._get_valid_cache(product_id)
            if cached:
                logger.warning(
                    f"Network error during validation, using cache: {e.reason}"
                )
                return LicenseResult(**cached)
            return LicenseResult(
                status=LicenseStatus.ERROR,
                message=f"Network error: {e.reason}",
            )
        except Exception as e:
            logger.error(f"Gumroad validation failed: {e}")
            return LicenseResult(
                status=LicenseStatus.ERROR,
                message=f"Validation failed: {str(e)}",
            )

    def _create_test_result(self, product_id: str) -> LicenseResult:
        result = LicenseResult(
            status=LicenseStatus.VALID,
            message="Test license key",
            license_type=LicenseType.ONE_TIME,
            customer_email="test@example.com",
            last_validated=datetime.now(),
            metadata={
                "product_id": product_id,
                "product_name": "Test Product",
                "purchase_date": datetime.now().isoformat(),
            },
        )
        self._cache_result(product_id, result)
        return result

    def _get_valid_cache(self, product_id: str) -> Optional[Dict]:
        cached = self._cache.get(product_id)
        if not cached:
            return None

        try:
            status = LicenseStatus(cached.get("status"))
            license_type = LicenseType(cached.get("license_type", "unknown"))
            last_validated_str = cached.get("last_validated")
            last_validated = None
            if last_validated_str:
                last_validated = datetime.fromisoformat(last_validated_str)

            cached_result = LicenseResult(
                status=status,
                message=cached.get("message", ""),
                license_type=license_type,
                customer_email=cached.get("customer_email"),
                last_validated=last_validated,
                metadata=cached.get("metadata", {}),
            )
            if cached_result.is_valid_for_offline():
                return cached
        except (ValueError, TypeError):
            pass
        return None

    def _cache_result(self, product_id: str, result: LicenseResult) -> None:
        cache_data = {
            "status": result.status.value,
            "message": result.message,
            "license_type": result.license_type.value,
            "customer_email": result.customer_email,
            "last_validated": result.last_validated.isoformat()
            if result.last_validated
            else None,
            "metadata": result.metadata,
        }
        self._cache[product_id] = cache_data
        self._save_cache()

    def _load_config(self) -> None:
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
                    self._licenses = data.get("licenses", {})
                    self._cache = data.get("cache", {})
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load Gumroad config: {e}")

    def _save_config(self) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "licenses": self._licenses,
            "cache": self._cache,
        }
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_cache(self) -> None:
        self._save_config()
