from typing import List
from abc import ABC, abstractmethod

class ChunkStrategy(ABC):
    @abstractmethod
    def split(self, text: str) -> List[str]:
        pass

