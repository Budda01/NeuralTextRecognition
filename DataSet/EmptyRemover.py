import os
from PIL import Image

def is_image_black(image_path):
    """Проверяет, является ли изображение полностью чёрным."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            extrema = img.getextrema()
            # Если все значения R, G и B равны (0, 0), то изображение чёрное
            return all(channel == (0, 0) for channel in extrema)
    except Exception as e:
        print(f"Ошибка при проверке изображения {image_path}: {e}")
        return False

def remove_black_images_from_folder(base_folder):
    """Удаляет полностью чёрные изображения из всех подпапок base_folder."""
    print("Начинаем проверку изображений на чёрный цвет...")
    count = 0
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if not file_name.lower().endswith(".png"):
                    continue
                file_path = os.path.join(folder_path, file_name)
                if is_image_black(file_path):
                    os.remove(file_path)
                    print(f"Удалено: {file_path}")
                    count += 1
    print(f"Удаление завершено. Удалено {count} изображений.")

if __name__ == "__main__":
    base_output_folder = "output_imag"
    remove_black_images_from_folder(base_output_folder)
