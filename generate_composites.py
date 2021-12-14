import argparse
from collections import defaultdict
import cv2
import distort
import glob
import json
import numpy as np
import os
import os.path as osp
import random
import tqdm

NUM_CLASSES_IN_EACH_COMPOSITE = 4
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
VIDEO_EXTENSIONS = ["mp4", "mov"]


def is_file_type(filepath, extensions):
    extension = filepath.split(".")[-1]
    return extension.lower() in extensions


def gather_files(directory, allowed_extensions):
    gathered_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if is_file_type(f, allowed_extensions):
                gathered_files.append(osp.join(root, f))

    return gathered_files


def select_random_video_frame(video_path):
    vid = cv2.VideoCapture(video_path)
    total_num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_id = random.randint(0, total_num_frames - 1)
    vid.set(1, random_frame_id)
    _, img = vid.read()

    return img


class CompositeGenerator:
    def __init__(
        self,
        objects_dir,
        background_dir,
        num_obj_per_composite,
    ):
        self.objects_dir = objects_dir
        self.class_id_to_paths = defaultdict(list)
        self.class_id_to_name = {}
        self.annotation_id = 0
        self.image_id = 0
        self.images = []
        self.annotations = []

        assert self.get_object_classes()
        num_classes = len(self.class_id_to_paths)
        print(f"Detected {num_classes} classes.")

        self.num_obj_per_composite = num_obj_per_composite

        # Find all candidate image or video files for select background imagery from
        self.background_sources = gather_files(
            background_dir, IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
        )
        assert len(self.background_sources) > 0, "No background sources found!"

    def get_object_classes(self):
        object_dirs = [
            d for d in glob.glob(osp.join(self.objects_dir, "*")) if osp.isdir(d)
        ]

        # Must have following format: {class_id}_{class_name}
        for d in object_dirs:
            basename = osp.basename(d)
            class_id, class_name = basename.split("_")[0], basename.split("_")[-1]
            if "_" not in basename or not class_id.isdigit():
                print(d, "is not named correctly!")
                return False

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

        return True

    def generate_composite(self, output_path):
        # Randomly select a background source
        random_vid_or_img = random.choice(self.background_sources)
        if is_file_type(random_vid_or_img, VIDEO_EXTENSIONS):
            bg_img = select_random_video_frame(random_vid_or_img)
        elif is_file_type(random_vid_or_img, IMAGE_EXTENSIONS):
            bg_img = random_vid_or_img
        else:
            raise RuntimeError(
                f"{random_vid_or_img} is neither a video or an image."
            )
        bg_height, bg_width = bg_img.shape[:2]

        mask = np.zeros([bg_height, bg_width, 4], dtype=np.uint8)  # BGRA foreground
        segmentation_mask = np.zeros(
            [bg_height, bg_width], dtype=np.float32  # pixel class ids
        )

        # Randomly select object classes that will appear in composite
        chosen_class_ids = [
            random.choice(list(self.class_id_to_paths.keys()))
            for _ in range(self.num_obj_per_composite)
        ]
        for num_objects, class_id in enumerate(chosen_class_ids):
            ret = False
            while not ret:
                random_obj_path = random.choice(self.class_id_to_paths[class_id])
                obj_img = cv2.imread(random_obj_path, cv2.IMREAD_UNCHANGED)
                obj_img = distort.rotate_object(obj_img)
                obj_img = distort.resize_by_dim_and_area(obj_img, bg_height, bg_width)
                ret, mask, segmentation_mask = distort.attempt_composite(
                    obj_img, mask, segmentation_mask, num_objects
                )

        # Overlay mask of objects on to background
        composite = bg_img.copy()
        mask_no_alpha, mask_alpha = mask[:, :, :3], mask[:, :, 3]
        composite[mask_alpha > 0] = mask_no_alpha[mask_alpha > 0]

        # Add noise to the composite
        composite = distort.random_gamma(composite, 3.5)
        composite = distort.random_blur(composite, 2)
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
        self.image_id += 1

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
            self.annotation_id += 1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates dataset.")
    parser.add_argument("objects_dir")
    parser.add_argument("background_dir")
    parser.add_argument("out_json_path")
    parser.add_argument("out_composites_dir")
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    cg = CompositeGenerator(
        objects_dir=args.objects_dir,
        background_dir=args.background_dir,
        num_obj_per_composite=4,
    )

    if not osp.isdir(args.out_composites_dir):
        os.mkdir(args.out_composites_dir)

    try:
        for idx in tqdm.trange(40):
            out_path = osp.join(args.out_composites_dir, f"{idx:06}.png")
            composite = cg.generate_composite(out_path)
            cv2.imwrite(out_path, composite)
    finally:
        cg.compile_coco_json(args.out_json_path)
