import numpy as np
import cv2
from abstract_distortion import AbstractDistortion


class HueDistortion(AbstractDistortion):
    name = "_hue"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["hue", "tint", "shade", "color shift"]
        self._borders = {
            "_DECREASE": {
                "_SMALL": (0, 10),
                "_MEDIUM": (10, 30),
                "_LARGE": (30, 60),
            },
            "_INCREASE": {
                "_SMALL": (0, -10),
                "_MEDIUM": (-10, -30),
                "_LARGE": (-30, -60),
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
        ]

        sentence = np.random.choice(template_sentences)
        return sentence

    def _generate_sample(self):
        keys_list, vals = self._sample_dicts_recursively(self._borders)
        return ":".join(keys_list), np.random.uniform(vals[0], vals[1])

    def __call__(self, image, original_image):
        text, hue_shift = self._generate_sample()
        text = self._generate_sentence(text)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = np.mod(hsv_image[:, :, 0] + hue_shift, 180).astype(np.uint8)
        distorted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return distorted_image, text, original_image
