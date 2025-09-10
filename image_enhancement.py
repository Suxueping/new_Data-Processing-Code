import os
import json
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import argparse


def load_config(config_path):
    """Load enhancement configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def apply_enhancement(image: Image.Image, enhance_params) -> Image.Image:
    """
    Apply specified enhancement to an image

    Parameters:
        image: Original PIL Image object
        enhance_params: Dictionary containing enhancement parameters

    Returns:
        Enhanced PIL Image object
    """
    enhanced_img = image.copy()

    # Apply brightness adjustment if specified
    if 'brightness' in enhance_params:
        enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = enhancer.enhance(enhance_params['brightness'])

    # Apply contrast adjustment if specified
    if 'contrast' in enhance_params:
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(enhance_params['contrast'])

    # Apply sharpness adjustment if specified
    if 'sharpness' in enhance_params:
        enhancer = ImageEnhance.Sharpness(enhanced_img)
        enhanced_img = enhancer.enhance(enhance_params['sharpness'])

    # Apply color adjustment if specified
    if 'color' in enhance_params:
        enhancer = ImageEnhance.Color(enhanced_img)
        enhanced_img = enhancer.enhance(enhance_params['color'])

    # Apply Gaussian blur if specified
    if 'blur_radius' in enhance_params and enhance_params['blur_radius'] > 0:
        enhanced_img = enhanced_img.filter(
            ImageFilter.GaussianBlur(radius=enhance_params['blur_radius'])
        )

    return enhanced_img


def process_dataset(image_dir: str, annotation_dir: str, output_root: str, config) -> None:
    """
    Batch process image dataset, generate enhanced versions and synchronize annotations

    Parameters:
        image_dir: Path to directory containing original images
        annotation_dir: Path to directory containing original annotations
        output_root: Root directory for saving enhanced results
        config: Enhancement configuration dictionary
    """
    # Create output directories
    for enhancement in config['enhancements']:
        alias = enhancement['alias']
        os.makedirs(os.path.join(output_root, f'images_{alias}'), exist_ok=True)
        os.makedirs(os.path.join(output_root, f'annotations_{alias}'), exist_ok=True)

    # Process all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for img_file in os.listdir(image_dir):
        # Skip non-image files
        if not img_file.lower().endswith(image_extensions):
            continue

        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        img_path = os.path.join(image_dir, img_file)
        annotation_path = os.path.join(annotation_dir, f"{base_name}.json")

        # Skip images without annotations
        if not os.path.exists(annotation_path):
            print(f"Skipping unannotated file: {img_file}")
            continue

        try:
            with Image.open(img_path) as img:
                # Preserve EXIF data and convert to RGB if necessary
                exif = img.info.get('exif')
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Generate all enhanced versions
                for enhancement in config['enhancements']:
                    alias = enhancement['alias']
                    params = enhancement['parameters']

                    enhanced_img = apply_enhancement(img, params)

                    # Save enhanced image
                    new_img_name = f"{base_name}_{alias}{ext}"
                    img_output_path = os.path.join(output_root, f'images_{alias}', new_img_name)
                    enhanced_img.save(img_output_path, exif=exif)

                    # Process and save annotation file
                    try:
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            ann_data = json.load(f)
                        ann_data['imagePath'] = new_img_name  # Update image path in annotation

                        new_ann_name = f"{base_name}_{alias}.json"
                        ann_output_path = os.path.join(output_root, f'annotations_{alias}', new_ann_name)
                        with open(ann_output_path, 'w', encoding='utf-8') as f:
                            json.dump(ann_data, f, ensure_ascii=False, indent=4)
                    except Exception as e:
                        print(f"Failed to process annotation {annotation_path}: {str(e)}")

            print(f"Processed: {img_file} (generated {len(config['enhancements'])} enhanced versions)")

        except Exception as e:
            print(f"Failed to process image {img_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Batch process images with various enhancements')
    parser.add_argument('--image-dir', required=True, help='Directory containing original images')
    parser.add_argument('--annotation-dir', required=True, help='Directory containing original JSON annotations')
    parser.add_argument('--output-root', required=True, help='Root directory for saving enhanced results')
    parser.add_argument('--config', required=True, help='Path to enhancement configuration JSON file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Validate configuration
    if 'enhancements' not in config:
        raise ValueError("Configuration file must contain 'enhancements' section")

    # Process dataset
    process_dataset(args.image_dir, args.annotation_dir, args.output_root, config)
    print("All image enhancement processing completed!")


if __name__ == "__main__":
    main()
