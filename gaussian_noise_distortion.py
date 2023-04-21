from abstract_distortion import AbstractDistortion
import numpy as np
import cv2


class GaussianNoiseDistortion(AbstractDistortion):
    name = "_gaussian_noise"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["graininess", "grain", "noise"]
        self._borders = {
            "_PLACEHOLDER": {
                "_SMALL": (5, 25),
                "_MEDIUM": (25, 45),
                "_LARGE": (45, 75),
            },
        }

    def _generate_sentence(self, text):
        tokens = text.split(":")
        action_token, magnitude_token = tokens

        distortion_synonyms = self._synonyms
        magnitude_synonyms = self._key_synonyms[magnitude_token]

        distortion = np.random.choice(distortion_synonyms)
        magnitude = np.random.choice(magnitude_synonyms)

        template_sentences = [
            f"{magnitude.capitalize()} {distortion} has been removed from the image.",
            f"The image has been made {magnitude}ly de-{distortion}y.",
            f"The image has undergone a {magnitude} removal of {distortion}.",
        ]

        sentence = np.random.choice(template_sentences)
        return sentence

    def _generate_sample(self):
        keys_list, vals = self._sample_dicts_recursively(self._borders)
        return ":".join(keys_list), np.random.uniform(vals[0], vals[1])

    def __call__(self, image, original_image):
        text, variance = self._generate_sample()
        text = self._generate_sentence(text)
        noise = np.zeros(image.shape, np.float32)
        cv2.randn(noise, 0, variance)
        distorted_image = cv2.add(image, noise, dtype=cv2.CV_8UC3)
        return distorted_image, text, original_image
