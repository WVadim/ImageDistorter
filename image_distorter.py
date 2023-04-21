from brightness_distortion import BrightnessDistortion
from contrast_distortion import ContrastDistortion
from color_balance_distortion import ColorBalanceDistortion
from gaussian_noise_distortion import GaussianNoiseDistortion
from rotation_distortion import RotationDistortion
from saturation_distortion import SaturationDistortion
from snp_noise_distortion import SaltAndPepperNoiseDistortion
from vibrance_distortion import VibranceDistortion
import random
from typing import Tuple


class ImageDistorter:
    def __init__(
        self,
        number_of_effects_range: Tuple[int, int] = (2, 4),
    ):
        self._effects = {
            "brightness": BrightnessDistortion(),
            "contrast": ContrastDistortion(),
            "saturation": SaturationDistortion(),
            "color_balance": ColorBalanceDistortion(),
            "gaussian_noise": GaussianNoiseDistortion(),
            "rotation": RotationDistortion(),
            "snp_noise": SaltAndPepperNoiseDistortion(),
            "vibrance": VibranceDistortion(),
        }
        self._number_of_effects_range = number_of_effects_range

    def __call__(self, image):
        number_of_effects = random.randint(
            self._number_of_effects_range[0], self._number_of_effects_range[1]
        )
        random_effects = random.sample(
            self._effects.keys(), k=number_of_effects
        )
        result_text = []
        distorted = image.copy()
        for effect in random_effects:
            distorted, text, image = self._effects[effect](distorted, image)

            result_text.append(text)
        return distorted, result_text, image
