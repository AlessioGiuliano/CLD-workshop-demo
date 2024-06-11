import boto3
import webcolors

import argparse
import os
import shutil

SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png"]
CATEGORIES_MAPPING_PATH = "./categories.csv"
CSV_SEPARATOR = ","

DEFAULT_ARGS = {
    "max_label_count": 10,
    "min_confidence": 55.0,
    "min_sharpness": 60,
    "max_image_count": 20
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Sort images in directories according to their detected label in Rekognition."
    )

    parser.add_argument(
        "images_dir", help="The directory containing the images to sort. It must exist."
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        help="The directory in which to store the output. It will be created if it doesn't exist. Default=./output",
        default=".",
    )
    
    default = DEFAULT_ARGS["max_image_count"]
    parser.add_argument(
        "--max_image_count",
        "-maxi",
        help=f"The maximum amount of images to get in the folder. Default={default}",
        default=default
    )

    default = DEFAULT_ARGS["max_label_count"]
    parser.add_argument(
        "--max_label_count",
        "-maxl",
        help=f"The maximum amount of labels to detect per image. Default={default}",
        type=int,
        default=default,
    )

    default = DEFAULT_ARGS["min_confidence"]
    parser.add_argument(
        "--min_confidence",
        "-minc",
        help=f"The minimum confidence for a label to be taken into account. Default={default}",
        type=float,
        default=default,
    )
    
    default = DEFAULT_ARGS["min_sharpness"]
    parser.add_argument(
        "--min_sharpness",
        "-mins",
        help=f"The minimum sharpness score for an image in order to be sorted. Images with scores below this threshold will be ignored. Default={default}",
        type=float,
        default=default
    )

    return parser.parse_args()


def run_rekognition(
    image_path: str,
    client: boto3.client,
    max_labels: int = DEFAULT_ARGS["max_label_count"],
    min_confidence: float = DEFAULT_ARGS["min_confidence"],
) -> dict:
    """
    Call the AWS Rekognition API to extract labels and properties
    from the input image.
    """
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        
    response = client.detect_labels(
        Image={"Bytes": image_bytes},
        MaxLabels=max_labels,
        MinConfidence=min_confidence,
        Features=[
            "GENERAL_LABELS", "IMAGE_PROPERTIES"
        ]
    )
        
    return response


def is_sharp(image_properties: dict, threshold: float = DEFAULT_ARGS["min_sharpness"]) -> bool:
    """
    Return True if the image has a sharpness score above the threshold,
    False otherwise.
    """    
    # Default at 100 in order to keep the image if there is no Sharpness score for some reason
    avg_sharpness = image_properties.get("Quality", {}).get("Sharpness", 100) 
        
    return avg_sharpness > threshold


def get_dominant_color_name(image_properties: dict) -> str:
    dominant_colors = {
        color.get("PixelPercent", 0): color.get("SimplifiedColor")
        for color in image_properties.get("DominantColors")
    }
    
    return dominant_colors[max(dominant_colors.keys())]


def get_label_category(label: str) -> str:
    with open(CATEGORIES_MAPPING_PATH, "r") as file:
        for line in file:
            if line.startswith(label):
                return line.split(CSV_SEPARATOR)[1]
            
    return None
    

def get_image_paths(images_dir: str, max: int=-1) -> list[str]:
    """
    Get a list of image paths from a directory.

    Only images with an accepted extension will be retrieved.
    """
    images = []
    for filename in os.listdir(images_dir):
        image_path = os.path.join(images_dir, filename)
        if os.path.isfile(image_path):
            ext = os.path.splitext(filename)
            if ext[1].lower().lstrip(".") in SUPPORTED_EXTENSIONS:
                images.append(image_path)
        if len(images) == max:
            return images

    return images


if __name__ == "__main__":
    args = parse_args()

    OUTPUT_DIR = args.output_dir

    client = boto3.client("rekognition")
    image_paths = get_image_paths(args.images_dir, args.max_image_count)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for image_path in image_paths:
        response = run_rekognition(image_path, client, args.max_label_count, args.min_confidence)
        properties = response.get("ImageProperties")
        labels = response.get("Labels")
        
        if not is_sharp(properties, args.min_sharpness):
            print(f"Image {image_path} is considered blurry, not classifying...")
            continue
        
        dominant_color = get_dominant_color_name(properties)

        labels_dir = os.path.join(OUTPUT_DIR, "labels")
        colors_dir = os.path.join(OUTPUT_DIR, "colors")

        image_name = os.path.basename(image_path)
        
        def make_category(_dir: str) -> None:
            os.makedirs(_dir, exist_ok=True)
            shutil.copy(image_path, os.path.join(_dir, image_name))

        make_category(os.path.join(colors_dir, dominant_color))
            
        for label in labels:
            # Use only labels that apply to the whole image.
            if not label.get("Instances"):
                label_category = get_label_category(label["Name"])
                path = os.path.join(labels_dir, label_category, label["Name"])
                make_category(path)
            
