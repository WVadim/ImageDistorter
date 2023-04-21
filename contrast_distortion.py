from abstract_distortion import AbstractDistortion
import numpy as np
import cv2


class ContrastDistortion(AbstractDistortion):
    name = "_contrast"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["contrast", "distinction"]
        self._borders = {
            "_INCREASE": {
                "_SMALL": (-20, 15),
                "_MEDIUM": (-50, 15),
                "_LARGE": (-80, 15),
            },
            "_DECREASE": {
                "_SMALL": (20, 15),
                "_MEDIUM": (50, 15),
                "_LARGE": (80, 15),
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
        return ":".join(keys_list), np.random.normal(vals[0], vals[1])

    def __call__(self, image, original_image):
        text, contrast_value = self._generate_sample()
        text = self._generate_sentence(text)
        distorted_image = cv2.addWeighted(image, 1 + contrast_value / 127, image, 0, 0)
        return distorted_image, text, original_image
