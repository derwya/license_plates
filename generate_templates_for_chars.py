import random
import os
from typing import Dict

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

TEMPLATES_DIR: str = './char_templates/'

CHAR_H: int = 40
CHAR_W: int = 40

SEED = 1412

def apply_contextual_noise(image: NDArray, flip_probability_factor: float = 0.25) -> NDArray:


    noisy_image = image.copy()
    height, width = image.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Get the 3x3 neighborhood
            neighborhood = image[y - 1:y + 2, x - 1:x + 2]
            center_pixel_value = image[y, x]

            if center_pixel_value == 0:  # Black pixel
                white_neighbors = np.sum(neighborhood == 255)
                flip_chance = (white_neighbors / 8.0) * flip_probability_factor
                if random.random() < flip_chance:
                    noisy_image[y, x] = 255
            else:  # White pixel
                black_neighbors = np.sum(neighborhood == 0)
                flip_chance = (black_neighbors / 8.0) * flip_probability_factor
                if random.random() < flip_chance:
                    noisy_image[y, x] = 0
    return noisy_image


def apply_blob_noise(image: NDArray, num_blobs: int = 3, max_radius: int = 3) -> NDArray:
    noisy_image = image.copy()
    height, width = image.shape

    for _ in range(random.randint(1, num_blobs)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, max_radius)
        color = random.choice([0, 255])
        cv.circle(noisy_image, (x, y), radius, color, -1)
    return noisy_image


def apply_salt_and_pepper_noise(image: NDArray, amount: float = 0.02) -> NDArray:
    noisy_image = image.copy()
    num_noise_pixels = int(amount * image.size)

    # Add salt (white pixels)
    for _ in range(num_noise_pixels // 2):
        y = random.randint(0, image.shape[0] - 1)
        x = random.randint(0, image.shape[1] - 1)
        noisy_image[y, x] = 255

    # Add pepper (black pixels)
    for _ in range(num_noise_pixels // 2):
        y = random.randint(0, image.shape[0] - 1)
        x = random.randint(0, image.shape[1] - 1)
        noisy_image[y, x] = 0
    return noisy_image


def generate_character_images_with_noise(num_templates_per_char: int = 20) -> None:

    characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    fonts: Dict[int, str] = {
        cv.FONT_HERSHEY_DUPLEX: "DUPLEX",
        cv.FONT_HERSHEY_SIMPLEX: "SIMPLEX",
        #cv.FONT_HERSHEY_TRIPLEX: "TRIPLEX"
    }

    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    print(f"Using output directory: {TEMPLATES_DIR}")

    total_generated_count = 0

    for char in characters:
        char_dir = os.path.join(TEMPLATES_DIR, char)
        os.makedirs(char_dir, exist_ok=True)

        print(f"Generating {num_templates_per_char} templates for '{char}'...")

        for i in range(num_templates_per_char):
            font_face = random.choice(list(fonts.keys()))
            temp_canvas_size = 200
            temp_canvas = np.zeros((temp_canvas_size, temp_canvas_size), dtype=np.uint8)
            font_scale = random.uniform(2.8, 3.2)
            thickness = random.randint(2, 3)

            (text_w, text_h), _ = cv.getTextSize(char, font_face, font_scale, thickness)
            origin_x = (temp_canvas_size - text_w) // 2
            origin_y = (temp_canvas_size + text_h) // 2
            cv.putText(temp_canvas, char, (origin_x, origin_y), font_face, font_scale, 255, thickness, cv.LINE_AA)

            angle = random.uniform(-8, 8)
            center = (temp_canvas_size // 2, temp_canvas_size // 2)
            rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            rotated_canvas = cv.warpAffine(temp_canvas, rotation_matrix, (temp_canvas_size, temp_canvas_size))

            y_coords, x_coords = np.where(rotated_canvas > 0)
            if len(y_coords) == 0: continue

            cropped_char = rotated_canvas[np.min(y_coords):np.max(y_coords) + 1, np.min(x_coords):np.max(x_coords) + 1]

            h, w = cropped_char.shape
            scale_w = random.uniform(0.85, 1.1)
            scale_h = random.uniform(0.85, 1.1)
            new_stretched_w = max(1, int(w * scale_w))
            new_stretched_h = max(1, int(h * scale_h))

            processed_char = cv.resize(cropped_char, (new_stretched_w, new_stretched_h), interpolation=cv.INTER_AREA)

            h, w = processed_char.shape
            scale = min(CHAR_W / w, CHAR_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv.resize(processed_char, (new_w, new_h), interpolation=cv.INTER_AREA)
            _, resized = cv.threshold(resized, 127, 255, cv.THRESH_BINARY)

            final_canvas = np.zeros((CHAR_H, CHAR_W), dtype=np.uint8)
            paste_x = (CHAR_W - new_w) // 2
            paste_y = (CHAR_H - new_h) // 2
            final_canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

            noisy_image = final_canvas.copy()
            noisy_image = apply_contextual_noise(noisy_image, flip_probability_factor=random.uniform(0.1, 0.4))

            # if random.random() < 0.2:
            #     noisy_image = apply_blob_noise(noisy_image, num_blobs=2, max_radius=2)

            if random.random() < 0.3:
                noisy_image = apply_salt_and_pepper_noise(noisy_image, amount=random.uniform(0.01, 0.03))

            font_name = fonts[font_face]
            filename = f"{font_name}_{i + 1}.png"
            filepath = os.path.join(char_dir, filename)
            cv.imwrite(filepath, noisy_image)
            total_generated_count += 1

    print(f"\n Successfully generated {total_generated_count} noisy character images in '{TEMPLATES_DIR}'")


if __name__ == '__main__':
    random.seed(SEED)
    generate_character_images_with_noise(num_templates_per_char=200)