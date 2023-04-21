from abstract_distortion import AbstractDistortion
import numpy as np
import cv2


class ColorBalanceDistortion(AbstractDistortion):
    name = "_color_balance"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["color adjustment", "color correction", "color equilibrium"]
        self._borders = {
            "_INCREASE": {
                "_SMALL": (0.93, 0.99),
                "_MEDIUM": (0.85, 0.93),
                "_LARGE": (0.75, 0.85),
            },
            "_DECREASE": {
                "_SMALL": (1.01, 1.07),
                "_MEDIUM": (1.07, 1.15),
                "_LARGE": (1.15, 1.25),
            },
        }

    def _generate_sentence(self, text):
        tokens = text.split(":")
        action_token, magnitude_token = tokens

        distortion_synonyms = self._synonyms
        action_synonyms = self._key_synonyms[action_token]
        magnitude_synonyms = self._key_synonyms[magnitude_token]

        distortion = np.random.choice(distortion_synonyms)
        action = np.random.choice(action_synonyms)
        magnitude = np.random.choice(magnitude_synonyms)

        template_sentences = [
            f"The {distortion} of the image has been {action}d by a {magnitude} amount.",
            f"A {magnitude} {action} in {distortion} has been applied to the image.",
            f"The image has undergone a {magnitude} {action} in {distortion}.",
            f"The image's {distortion} has been {action}d, resulting in a {magnitude} change.",
            f"{magnitude.capitalize()} {action} of {distortion} has been applied to the image."
        ]

        sentence = np.random.choice(template_sentences)
        return sentence

    def _generate_sample(self):
        keys_list, vals = self._sample_dicts_recursively(self._borders)
        return ":".join(keys_list), np.random.uniform(vals[0], vals[1], size=3)

    def __call__(self, image, original_image):
        text, balance_ratio = self._generate_sample()
        text = self._generate_sentence(text)
        distorted_image = np.float32(image)
        distorted_image *= balance_ratio
        distorted_image = np.clip(image, 0, 255).astype(np.uint8)
        return distorted_image, text, original_image
