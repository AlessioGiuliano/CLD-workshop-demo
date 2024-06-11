import boto3
from PIL import Image

import argparse
import os

from classify import run_rekognition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", help="The path to the image to crop.")

    parser.add_argument(
        "--output_dir",
        "-o",
        help="The directory to store the cropped images. Default=.",
        default=".",
    )

    return parser.parse_args()


def extract_bounding_boxes(
    labels: dict, min_confidence: float = 75.0
) -> dict[str:dict]:
    """
    Extract bounding boxes for each label.
    Return a dict in the form:
    {
        <image_subname>: <bounding_box>
    }
    """
    result = {}

    for label in labels:
        name = label["Name"]

        for i, instance in enumerate(label.get("Instances", [])):
            if instance["Confidence"] >= min_confidence:
                result[f"{name}-{i}"] = instance["BoundingBox"]

    return result


def make_cropped_images(
    image_path: str, bounding_boxes: dict[str:dict], output_dir: str
) -> None:
    """
    Crop and write the new images to the disk.
    """
    image_extension = os.path.splitext(image_path)[1]
    image = Image.open(image_path)
    width = image.width
    height = image.height

    for name, bbox in bounding_boxes.items():
        left = int(bbox["Left"] * width)
        top = int(bbox["Top"] * height)
        right = left + int(bbox["Width"] * width)
        bottom = top + int(bbox["Height"] * height)

        path = os.path.join(output_dir, name + image_extension)

        cropped = image.crop((left, top, right, bottom))
        cropped.save(path)


if __name__ == "__main__":
    args = parse_args()

    client = boto3.client("rekognition")
    response = run_rekognition(args.input_path, client)

    labels = response.get("Labels")

    bounding_boxes = extract_bounding_boxes(labels)

    os.makedirs(args.output_dir, exist_ok=True)
    make_cropped_images(args.input_path, bounding_boxes, args.output_dir)
