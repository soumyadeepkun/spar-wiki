from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class RetrievalEngine(ABC):
    @abstractmethod
    def search(
        self,
        term_ids: Sequence[int],
        term_weights: Sequence[float] | None = None,
        k: int = 10,
        min_score: float | None = None,
    ) -> list[object]:
        raise NotImplementedError
