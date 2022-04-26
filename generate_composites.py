import glob
import json
import os
import os.path as osp
import random
from collections import defaultdict

import cv2
import distort
import numpy as np

NUM_CLASSES_IN_EACH_COMPOSITE = 4
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
VIDEO_EXTENSIONS = ["mp4", "mov"]


def is_file_type(filepath, extensions):
    """Determines if the given filepath has an extension in the given list of
    extensions"""
    extension = filepath.split(".")[-1]
    return extension.lower() in extensions


def gather_files(directory, allowed_extensions):
    """
    Returns a list of files in the given dir and all its sub-dirs that have an extension
    in the given list of allowed extensions
    :param directory: path to directory to os.walk through
    :param allowed_extensions: list of extensions
    :return: list of paths to files with an allowed extension
    """
    gathered_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if is_file_type(f, allowed_extensions):
                gathered_files.append(osp.join(root, f))

    return gathered_files


def select_random_video_frame(video_path):
    """Returns a random frame (np.uint8) from the video located at video_path"""
    vid = cv2.VideoCapture(video_path)
    total_num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_id = random.randint(0, total_num_frames - 1)
    vid.set(1, random_frame_id)
    _, img = vid.read()

    return img


class CompositeGenerator:
    def __init__(
        self, objects_dir, background_dir, num_obj_per_composite, existing_data=None
    ):
        self.objects_dir = objects_dir
        self.class_id_to_paths = defaultdict(list)
        self.class_id_to_name = {}

        if existing_data is not None:
            print("Loading data from existing json...")
            self.images = existing_data["images"]
            self.annotations = existing_data["annotations"]
        else:
            self.images = []
            self.annotations = []

        self.get_object_classes()
        num_classes = len(self.class_id_to_paths)
        assert num_classes > 0, "No object folders were found!"
        print(f"Detected {num_classes} classes.")

        self.num_obj_per_composite = num_obj_per_composite

        # Find all candidate image or video files for select background imagery from
        self.background_sources = gather_files(
            background_dir, IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
        )
        assert len(self.background_sources) > 0, "No background sources found!"

    @property
    def image_id(self):
        return len(self.images)

    @property
    def annotation_id(self):
        return len(self.annotations)

    def get_object_classes(self):
        object_dirs = [
            d for d in glob.glob(osp.join(self.objects_dir, "*")) if osp.isdir(d)
        ]

        # Must have following format: {class_id}_{class_name}
        for d in object_dirs:
            basename = osp.basename(d)
            class_id = basename.split("_")[0]
            class_name = basename[len(class_id + "_") :]
            if "_" not in basename or not class_id.isdigit():
                print(d, "is not named correctly; ignoring it.")
                continue

            obj_img_paths = []
            for root, _, files in os.walk(d):
                obj_img_paths.extend(
                    [
                        osp.join(root, f)
                        for f in files
                        if is_file_type(f, IMAGE_EXTENSIONS)
                    ]
                )

            class_id = int(class_id)
            self.class_id_to_paths[class_id].extend(obj_img_paths)
            self.class_id_to_name[class_id] = class_name

    def generate_composite(self, output_path):
        # Randomly select a background source
        random_vid_or_img = random.choice(self.background_sources)
        if is_file_type(random_vid_or_img, VIDEO_EXTENSIONS):
            bg_img = select_random_video_frame(random_vid_or_img)
        elif is_file_type(random_vid_or_img, IMAGE_EXTENSIONS):
            bg_img = cv2.imread(random_vid_or_img)
        else:
            raise RuntimeError(f"{random_vid_or_img} is neither a video or an image.")
        bg_height, bg_width = bg_img.shape[:2]

        segmentation_mask = np.zeros(
            [bg_height, bg_width], dtype=np.float32  # pixel class ids
        )

        # Randomly select object classes that will appear in composite
        chosen_class_ids = [
            random.choice(list(self.class_id_to_paths.keys()))
            for _ in range(self.num_obj_per_composite)
        ]
        masks = []
        for num_objects, class_id in enumerate(chosen_class_ids):
            ret = False
            while not ret:
                random_obj_path = random.choice(self.class_id_to_paths[class_id])
                obj_img = cv2.imread(random_obj_path, cv2.IMREAD_UNCHANGED)
                obj_img = distort.rotate_object(obj_img)
                obj_img = distort.random_flip(obj_img)
                for attempt in range(10):
                    obj_img = distort.resize_by_dim(obj_img, bg_height, bg_width)
                    ret, masks, segmentation_mask = distort.attempt_composite(
                        obj_img, masks, segmentation_mask, bg_height, bg_width
                    )
                    if ret:
                        break

        # Overlay mask of objects on to background with alpha blending
        composite = bg_img.copy()
        for mask in masks:
            composite = distort.alpha_blend(composite, mask)

        # Add noise to the composite
        composite = distort.random_gamma(composite, 1.7)
        composite = distort.random_blur(composite, 3)
        # composite = distort.random_noise(composite)

        self.segmentation_mask_to_coco(segmentation_mask, chosen_class_ids)
        self.images.append(
            {
                "id": self.image_id,
                "file_name": osp.basename(output_path),
                "height": bg_height,
                "width": bg_width,
            }
        )

        return composite

    def segmentation_mask_to_coco(self, segmentation_mask, chosen_class_ids):
        height, width = segmentation_mask.shape[:2]
        for idx, class_id in enumerate(chosen_class_ids):
            mask = np.zeros([height, width], dtype=np.uint8)
            mask[segmentation_mask == idx + 1] = 255
            polygons, area = distort.get_poly_from_mask(mask)
            x, y, w, h = cv2.boundingRect(mask)

            self.annotations.append(
                {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": class_id,
                    "segmentation": polygons,
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }
            )

    def compile_coco_json(self, output_path):
        coco_dict = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [
                {"id": class_id, "name": class_name, "supercategory": "null"}
                for class_id, class_name in self.class_id_to_name.items()
            ],
        }
        with open(output_path, "w") as f:
            json.dump(coco_dict, f, indent=4)
