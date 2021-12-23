import argparse
import tqdm
from generate_composites import CompositeGenerator
import os
import os.path as osp
import json
import cv2
import signal

# Global Boolean variable that indicates that a signal has been received
interrupted = False
def signal_handler(signum, frame):
    global interrupted
    interrupted = True


def main():
    parser = argparse.ArgumentParser(description="Generates dataset.")
    parser.add_argument("objects_dir")
    parser.add_argument("background_dir")
    parser.add_argument("out_json_path")
    parser.add_argument("out_composites_dir")
    parser.add_argument("num_composites", type=int)
    parser.add_argument("-p", "--prefix", default='')
    args = parser.parse_args()

    # Register the signal handler (for handling slurm job cancellation)
    signal.signal(signal.SIGTERM, signal_handler)

    # Prefix (if provided) to prepend to all generated image names and json name
    prefix = args.prefix + "_" if args.prefix != "" else ""
    j_basename = osp.basename(args.out_json_path)
    json_path = args.out_json_path.replace(j_basename, prefix + j_basename)

    if not osp.isdir(args.out_composites_dir):
        os.mkdir(args.out_composites_dir)

    # Resume from existing json if it exists
    if osp.isfile(json_path):
        with open(json_path) as f:
            existing_data = json.load(f)
        num_existing = len(existing_data["images"])
    else:
        existing_data = None
        num_existing = 0

    cg = CompositeGenerator(
        objects_dir=args.objects_dir,
        background_dir=args.background_dir,
        num_obj_per_composite=4,
        existing_data=existing_data,
    )

    try:
        for idx in tqdm.trange(num_existing, args.num_composites):
            out_path = osp.join(args.out_composites_dir, prefix + f"{idx:06}.png")
            composite = cg.generate_composite(out_path)
            cv2.imwrite(out_path, composite)
            if interrupted:
                break
    finally:
        cg.compile_coco_json(json_path)


if __name__ == "__main__":
    main()
