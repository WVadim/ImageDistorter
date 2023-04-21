import numpy as np
import cv2
from abstract_distortion import AbstractDistortion


class JPEGCompressionDistortion(AbstractDistortion):
    name = "_jpeg_compression"

    def __init__(self):
        super().__init__()
        self._synonyms = ["artifacts"]
        self._borders = {
            "_SMALL": (75, 95),
            "_MEDIUM": (40, 70),
            "_LARGE": (10, 35),
        }

    def _generate_sentence(self, text):
        tokens = text.split(":")
        magnitude_token = tokens[0]

        distortion_synonyms = self._synonyms
        magnitude_synonyms = self._key_synonyms[magnitude_token]

        distortion = np.random.choice(distortion_synonyms)
        magnitude = np.random.choice(magnitude_synonyms)

        template_sentences = [
            f"The image has been cleaned from {magnitude} {distortion}.",
            f"{magnitude.capitalize()} {distortion} has been removed from the image.",
        ]

        sentence = np.random.choice(template_sentences)
        return sentence

    def _generate_sample(self):
        keys_list, vals = self._sample_dicts_recursively(self._borders)
        return ":".join(keys_list), np.random.randint(vals[0], vals[1])

    def __call__(self, image, original_image):
        text, quality = self._generate_sample()
        text = self._generate_sentence(text)
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        distorted_image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        return distorted_image, text, original_image
