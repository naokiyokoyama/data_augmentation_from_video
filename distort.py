import math
import random
from collections import Counter

import cv2
import numpy as np


def random_gamma(image, max_gamma):
    gamma = 1 + (max_gamma - 1) * np.random.random()
    rand = np.random.random()
    if rand > 0.5:
        invGamma = gamma
    else:
        invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def random_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 200
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    noisy = image + gauss
    return noisy


def random_blur(img, maxBlur):
    blur_kernel = 1 + int(np.random.random() * maxBlur)
    img = cv2.blur(img, (blur_kernel, blur_kernel))
    return img


def random_flip(img):
    if np.random.rand() > 0.5:
        return cv2.flip(img, 1)
    return img


def crop_box(img):  # Input: Image with only the object
    alpha = img[:, :, 3]
    x, y, w, h = cv2.boundingRect(alpha)
    return img[y : y + h, x : x + w]


def rotate_object(img):
    height, width = img.shape[:2]
    hypotenuse = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    blank_image = np.zeros([hypotenuse, hypotenuse, 4], np.uint8)
    x_offset = (hypotenuse - width) // 2
    y_offset = (hypotenuse - height) // 2
    blank_image[y_offset : y_offset + height, x_offset : x_offset + width] = img
    rotation_degrees = np.random.rand() * 360
    M = cv2.getRotationMatrix2D((hypotenuse / 2, hypotenuse / 2), rotation_degrees, 1)
    dst = cv2.warpAffine(blank_image, M, (hypotenuse, hypotenuse))
    dst = crop_box(dst)

    return dst


def resize_by_dim(
    object_img,
    bg_height,
    bg_width,
    lower_dim_bound=0.15,
    upper_dim_bound=0.95,
):
    obj_height, obj_width = object_img.shape[:2]
    dim_scale = random.uniform(lower_dim_bound, upper_dim_bound)
    if obj_height > obj_width:
        new_obj_height = dim_scale * bg_height
        scale = new_obj_height / obj_height
    else:
        new_obj_width = dim_scale * bg_width
        scale = new_obj_width / obj_width

    resized = cv2.resize(
        object_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    return resized


def attempt_composite(
    obj_img, masks, segmentation_mask, height, width, occlusion_thresh=0.4
):
    """
    :param obj_img: the BGRA object
    :param masks: BGRA cv2 uint8 image, a black image with BGRA objects added in
    :param segmentation_mask: single channel image showing what pixels in mask belong to
     what object class
    :param height:
    :param width:
    :param occlusion_thresh: max allowable occlusion
    :return:
    """
    obj_height, obj_width = obj_img.shape[:2]
    max_y_coord = height - obj_height + 1
    max_x_coord = width - obj_width + 1
    if max_y_coord < 1 or max_x_coord < 1:
        # The given obj_img is too large to fit
        return False, masks, segmentation_mask
    # Black out pixels in object img that don't have an alpha channel (just in case)
    obj_img[obj_img[:, :, 3] == 0] = (0, 0, 0, 0)
    num_objects = len(masks)
    pixel_counts = Counter()
    for mask_idx, mask in enumerate(masks):
        alpha = mask[:, :, 3]
        pixel_counts[mask_idx] = np.count_nonzero(alpha)

    success = True
    # Create a black BGRA mask the size of the composite with the object at a
    # random position
    candidate_mask = np.zeros([height, width, 4], dtype=np.uint8)
    ymin = np.random.randint(max_y_coord)
    xmin = np.random.randint(max_x_coord)
    candidate_mask[ymin : ymin + obj_height, xmin : xmin + obj_width] = obj_img
    candidate_alpha = candidate_mask[:, :, 3]

    if num_objects > 0:
        # Overlay the new object, and see if the new pixel counts indicate whether
        # any of the objects have become over-occluded
        candidate_segmentation_mask = segmentation_mask.copy()
        candidate_segmentation_mask[candidate_alpha > 0] = num_objects + 1
        for obj_idx in range(num_objects):
            new_pixel_count = len(
                candidate_segmentation_mask[candidate_segmentation_mask == obj_idx + 1]
            )
            if new_pixel_count < (1 - occlusion_thresh) * pixel_counts[obj_idx]:
                success = False
                break

    if success:
        masks.append(candidate_mask)
        segmentation_mask[candidate_alpha > 0] = num_objects + 1
        return True, masks, segmentation_mask

    return False, masks, segmentation_mask


def get_poly_from_mask(mask):
    """Returns masks in poly format, and its area"""
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = 0
    polygons = []
    for c in cnt:
        area += cv2.contourArea(c)
        polygons.append(np.ravel(c).tolist())

    return polygons, area


def alpha_blend(img, bgra_img):
    # Convert uint8 to float
    img_f = img.astype(float)
    bgra_img_f = bgra_img.astype(float)
    bgr_img_f, alpha_f = bgra_img_f[:, :, :3], bgra_img_f[:, :, 3:]
    alpha_f /= 255.0

    # Multiply the background with ( 1 - alpha )
    background = (1.0 - alpha_f) * img_f
    # Multiply the foreground with the alpha matte
    foreground = alpha_f * bgr_img_f

    # Add the masked foreground and background.
    out_img = cv2.add(foreground, background).astype(np.uint8)

    return out_img
