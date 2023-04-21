from abstract_distortion import AbstractDistortion
import numpy as np
import cv2


class SaltAndPepperNoiseDistortion(AbstractDistortion):
    name = "_salt_and_pepper_noise"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["speckle noise", "speckles", "spotted pattern"]
        self._borders = {
            "_PLACEHOLDER": {
                "_SMALL": (0.02, 0.05),
                "_MEDIUM": (0.05, 0.08),
                "_LARGE": (0.08, 0.11),
            },
        }
        self._salt_and_pepper_ratio = 0.5

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
        text, amount = self._generate_sample()
        text = self._generate_sentence(text)

        h, w, c = image.shape

        s_vs_p = self._salt_and_pepper_ratio

        sp_img = image.copy()
        num_salt = np.ceil(amount * image.size * s_vs_p)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))

        # Generate salt noise
        y_salt = np.random.randint(0, h, int(num_salt))
        x_salt = np.random.randint(0, w, int(num_salt))
        c_salt = np.random.randint(0, c, int(num_salt))
        sp_img[y_salt, x_salt, c_salt] = 255

        # Generate pepper noise
        y_pepper = np.random.randint(0, h, int(num_pepper))
        x_pepper = np.random.randint(0, w, int(num_pepper))
        c_pepper = np.random.randint(0, c, int(num_pepper))
        sp_img[y_pepper, x_pepper, c_pepper] = 0

        return sp_img, text, original_image
