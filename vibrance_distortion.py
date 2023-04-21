from abstract_distortion import AbstractDistortion
import numpy as np
import cv2


class VibranceDistortion(AbstractDistortion):
    name = "_vibrance"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["vibrancy", "colorfulness", "color intensity"]
        self._borders = {
            "_INCREASE": {
                "_SMALL": (0.85, 1.0),
                "_MEDIUM": (0.65, 0.85),
                "_LARGE": (0.5, 0.65),
            },
            "_DECREASE": {
                "_SMALL": (1.0, 1.15),
                "_MEDIUM": (1.15, 1.3),
                "_LARGE": (1.3, 1.5),
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
        text, vibrance = self._generate_sample()
        print(vibrance)
        text = self._generate_sentence(text)
        distorted_image = np.float32(image)
        hsv_image = cv2.cvtColor(
            np.clip(distorted_image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV
        )
        h, s, v = cv2.split(hsv_image)
        s = (s.astype(np.float32) * vibrance).clip(0, 255).astype(np.uint8)
        hsv_image = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), text, original_image

