from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class LicenseStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"
    ERROR = "error"


class LicenseType(Enum):
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    UNKNOWN = "unknown"


@dataclass
class LicenseResult:
    status: LicenseStatus
    message: str = ""
    license_type: LicenseType = LicenseType.UNKNOWN
    expires_at: datetime | None = None
    customer_email: str | None = None
    last_validated: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(tz=timezone.utc) >= self.expires_at

    def is_valid_for_offline(self) -> bool:
        if self.status != LicenseStatus.VALID:
            return False
        if self.is_expired():
            return False
        if self.license_type == LicenseType.ONE_TIME:
            return True
        if self.license_type == LicenseType.SUBSCRIPTION:
            if not self.last_validated:
                return False
            grace_period = timedelta(days=30)
            return (
                datetime.now(tz=timezone.utc) - self.last_validated
                < grace_period
            )
        return False


class LicenseProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass

    @abstractmethod
    def validate(self, config: dict[str, Any]) -> LicenseResult:
        pass
