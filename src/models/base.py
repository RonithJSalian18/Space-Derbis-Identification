from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple, Dict, Any


class BaseModelBuilder(ABC):
    """Abstract Base Class for all space debris classification model builders."""

    def __init__(self, input_shape: Tuple[int, int, int], config: Dict[str, Any] = None):
        self.input_shape = input_shape
        self.config = config or {}

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Constructs and returns an uncompiled tf.keras.Model instance."""
        pass
