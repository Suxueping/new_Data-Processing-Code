import os
import json
import argparse
from tqdm import tqdm


def load_classes(classes_file=None, classes_str=None):
    """
    Load class list from either a file or a string

    Parameters:
        classes_file: Path to file containing classes (one per line)
        classes_str: Comma-separated string of classes

    Returns:
        List of class names
    """
    if classes_file:
        with open(classes_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif classes_str:
        return [cls.strip() for cls in classes_str.split(',') if cls.strip()]
    else:
        raise ValueError("Either --classes-file or --classes must be specified")


def convert_json_to_txt(json_dir: str, save_dir: str, classes: List[str], format: str = "yolo") -> None:
    """
    Convert JSON annotations to TXT format (supports YOLO and VOC styles)

    Parameters:
        json_dir: Directory containing JSON annotation files
        save_dir: Directory to save TXT files
        classes: List of class names (matching labels in annotations)
        format: Output format ('yolo' or 'voc')
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Process all JSON files
    for json_fname in tqdm(os.listdir(json_dir), desc="Converting annotation files"):
        if not json_fname.lower().endswith('.json'):
            continue

        json_path = os.path.join(json_dir, json_fname)
        try:
            # Read JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get image dimensions
            img_h = data.get('imageHeight')
            img_w = data.get('imageWidth')

            if not img_h or not img_w:
                raise ValueError("Image dimensions not found in JSON")

            # Generate TXT content
            txt_content = []
            for shape in data.get('shapes', []):
                label = shape.get('label')
                if not label:
                    continue

                if label not in classes:
                    raise ValueError(f"Unknown class '{label}' found in annotations")

                # Get coordinates based on format
                points = shape.get('points', [])
                if not points:
                    continue

                class_idx = classes.index(label)

                if format == 'yolo':
                    # YOLO format: normalized coordinates (class x1 y1 x2 y2 ...)
                    normalized = [f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in points]
                    txt_line = f"{class_idx} {' '.join(normalized)}\n"
                else:  # voc format
                    # VOC format: absolute coordinates (class x1 y1 x2 y2 ...)
                    coordinates = [f"{p[0]:.1f} {p[1]:.1f}" for p in points]
                    txt_line = f"{label} {' '.join(coordinates)}\n"

                txt_content.append(txt_line)

            # Save TXT file
            txt_fname = os.path.splitext(json_fname)[0] + '.txt'
            with open(os.path.join(save_dir, txt_fname), 'w', encoding='utf-8') as f:
                f.writelines(txt_content)

        except UnicodeDecodeError:
            print(f"Skipping file with encoding issues: {json_path}")
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {json_path}")
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Convert JSON annotations to TXT format')
    parser.add_argument('--json-dir', required=True, help='Directory containing JSON annotation files')
    parser.add_argument('--save-dir', required=True, help='Directory to save converted TXT files')
    parser.add_argument('--classes-file', help='File containing class names (one per line)')
    parser.add_argument('--classes', help='Comma-separated list of class names (e.g., "class1,class2")')
    parser.add_argument('--format', default='yolo', choices=['yolo', 'voc'],
                        help='Output format (yolo or voc, default: yolo)')

    args = parser.parse_args()

    # Validate input
    if not args.classes_file and not args.classes:
        parser.error("Either --classes-file or --classes must be specified")

    # Load classes
    classes_list = load_classes(args.classes_file, args.classes)
    print(f"Loaded {len(classes_list)} classes for conversion")

    # Perform conversion
    convert_json_to_txt(args.json_dir, args.save_dir, classes_list, args.format)
    print("Annotation format conversion completed!")


if __name__ == "__main__":
    main()
