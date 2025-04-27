import os
from PIL import Image, ImageDraw, ImageFont
import math

def generate_letters_with_fonts(font_files, output_dir, font_size, image_size=(960, 540), spacing=5):
    """
    Generates images for each letter (upper, lower case, and punctuation) with different fonts, displaying the same letter multiple times per font in one image.
    """
    # All letters and symbols to be generated
    upper_case_letters = list(
        "I")
    lower_case_letters = list(
        "")
    punctuation = list("")

    all_characters = upper_case_letters + lower_case_letters + punctuation

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate image for each character
    for class_index, character in enumerate(all_characters, start=1):
        image = Image.new('RGB', image_size, 'black')
        draw = ImageDraw.Draw(image)

        # Adjust font size dynamically
        adjusted_font_size = font_size + 6 # Increase the base font size

        # Calculate rows and columns for the grid
        repeat_count = 2  # Repeat the character 5 times per font
        total_fonts = len(font_files)
        total_cells = total_fonts * repeat_count
        cols = 10
        rows = math.ceil(total_cells / cols)

        # Calculate dynamic cell size
        cell_width = (image_size[0] - spacing * (cols + 1)) // cols
        cell_height = (image_size[1] - spacing * (rows + 1)) // rows

        # Draw each font with repeated characters
        for idx, font_path in enumerate(font_files):
            # Load font
            try:
                pillow_font = ImageFont.truetype(font_path, adjusted_font_size)
            except Exception as e:
                print(f"Error loading font: {font_path} - {e}")
                continue

            for repeat_idx in range(repeat_count):
                global_idx = idx * repeat_count + repeat_idx
                row = global_idx // cols
                col = global_idx % cols

                # Position in grid
                x = spacing + col * (cell_width + spacing)
                y = spacing + row * (cell_height + spacing)

                # Center character in its grid cell
                text_bbox = draw.textbbox((0, 0), character, font=pillow_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x + (cell_width - text_width) // 2
                text_y = y + (cell_height - text_height) // 2

                draw.text((text_x, text_y), character, font=pillow_font, fill='white')

        # Save the output image
        output_file = os.path.join(output_dir, f"{class_index}.png")
        image.save(output_file)
        print(f"Saved: {output_file}")

# Example usage
if __name__ == "__main__":
    import glob

    font_files = glob.glob("fonts_folder/*.otf")  # Adjust font path as needed
    output_dir = "output_images"

    # Ensure there are fonts to process
    if len(font_files) > 0:
        generate_letters_with_fonts(font_files, output_dir, font_size=15)
    else:
        print("No fonts found in the specified folder.")