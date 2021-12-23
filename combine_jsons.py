import json
import argparse
import glob
import os.path as osp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()

    all_json_files = glob.glob(osp.join(args.json_dir, "*.json"))

    # Remove output json from consideration if it already exists
    if args.output_json in all_json_files:
        all_json_files.pop(all_json_files.index(args.output_json))

    image_id_offset = 0
    annotation_id_offset = 0
    combined_data = {
        "images": [],
        "annotations": [],
    }
    for json_idx, json_file in enumerate(sorted(all_json_files)):
        with open(json_file) as f:
            data = json.load(f)
        print(
            f"{json_file} has {len(data['images'])} images, "
            f"{len(data['annotations'])} annotations"
        )

        if json_idx > 0:
            for idx, image in enumerate(data["images"]):
                data["images"][idx]["id"] += image_id_offset
            for idx, annotation in enumerate(data["annotations"]):
                data["annotations"][idx]["id"] += annotation_id_offset
                data["annotations"][idx]["image_id"] += image_id_offset
        else:
            # Assume all jsons have same categories; just steal from first one
            combined_data["categories"] = data["categories"]

        combined_data["images"].extend(data["images"])
        combined_data["annotations"].extend(data["annotations"])

        # Update offsets
        image_id_offset = len(combined_data["images"])
        annotation_id_offset = len(combined_data["annotations"])

    print("Saving combined json...")  # may take a bit of time, eases anxiety
    with open(args.output_json, "w") as f:
        json.dump(combined_data, f)
    print(
        f"{args.output_json} has {image_id_offset} images, "
        f"{annotation_id_offset} annotations!!"
    )


if __name__ == "__main__":
    main()
