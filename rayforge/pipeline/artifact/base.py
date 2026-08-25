from __future__ import annotations

from abc import ABC, abstractmethod

from .handle import BaseArtifactHandle


class BaseArtifact(ABC):
    @property
    def artifact_type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def build_handle(self, key: str) -> BaseArtifactHandle:
        pass
