from abstract_distortion import AbstractDistortion
import numpy as np
import cv2
import math


class RotationDistortion(AbstractDistortion):
    name = "_rotation"

    def __init__(
        self,
    ):
        super().__init__()
        self._synonyms = ["rotation", "spin", "twist", "turn"]
        self._borders = {
            "_RIGHT": {
                "_SMALL": (5, 2.5),
                "_MEDIUM": (10, 2.5),
                "_LARGE": (15, 2.5),
            },
            "_LEFT": {
                "_SMALL": (-5, 2.5),
                "_MEDIUM": (-10, 2.5),
                "_LARGE": (-15, 2.5),
            },
        }

    @staticmethod
    def _largest_rotated_rect(w, h, angle):
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return bb_w - 2 * x, bb_h - 2 * y

    def _generate_sentence(self, text):
        tokens = text.split(":")
        action_token, magnitude_token = tokens

        distortion_synonyms = self._synonyms
        action_synonyms = self._key_synonyms[action_token]
        magnitude_synonyms = self._key_synonyms[magnitude_token]

        distortion = np.random.choice(distortion_synonyms)
        action = np.random.choice(action_synonyms)
        magnitude = np.random.choice(magnitude_synonyms)

        if distortion in ["rotation", "spin", "twist", "turn"]:
            template_sentences = [
                f"The image has been {action} {distortion}ed by a {magnitude} angle.",
                f"A {magnitude} {action} {distortion} has been applied to the image.",
                f"The image has undergone a {magnitude} {action} {distortion}.",
            ]
        else:
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

    @staticmethod
    def _crop_around_center(image, width, height):
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if width > image_size[0]:
            width = image_size[0]

        if height > image_size[1]:
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def __call__(self, image, original_image):
        text, angle = self._generate_sample()
        text = self._generate_sentence(text)
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix(
            [
                [1, 0, int(new_w * 0.5 - image_w2)],
                [0, 1, int(new_h * 0.5 - image_h2)],
                [0, 0, 1],
            ]
        )

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR
        )

        rect = self._largest_rotated_rect(
            image_size[0], image_size[1], math.radians(angle)
        )

        result_cropped = self._crop_around_center(result, *rect)
        image_cropped = self._crop_around_center(original_image, *rect)

        return result_cropped, text, image_cropped
