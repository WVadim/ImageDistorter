from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple


class AbstractDistortion(ABC):
    name = "_abstract_distortion"

    def __init__(self, **kwargs):
        self._borders = {}
        self._synonyms = []
        self._key_synonyms = {
            "_INCREASE": ["increase", "boost", "rise", "enhance"],
            "_DECREASE": ["decrease", "reduce", "lower", "diminish"],
            "_LEFT": ["left", "counterclockwise", "anti-clockwise"],
            "_RIGHT": ["right", "clockwise"],
            "_SMALL": ["small", "slight", "minimal", "mild"],
            "_MEDIUM": ["medium", "moderate", "average", "fair"],
            "_LARGE": ["large", "significant", "substantial", "considerable"],
        }

    @staticmethod
    def _sample_dicts_recursively(dictionary: Dict[str, Any]) -> Tuple[List[str], Any]:
        samples = []
        while isinstance(dictionary, dict):
            choice = np.random.choice(list(dictionary.keys()))
            samples.append(choice)
            dictionary = dictionary[choice]
        return samples, dictionary

    @abstractmethod
    def _generate_sentence(self, text):
        raise NotImplementedError(
            "AbstractDistortion.generate_sentence() is not implemented"
        )

    @abstractmethod
    def _generate_sample(self):
        raise NotImplementedError("AbstractDistortion._generate() is not implemented")

    @abstractmethod
    def __call__(self, image, original_image):
        raise NotImplementedError("AbstractDistortion.__call__() is not implemented")
