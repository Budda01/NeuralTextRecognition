import os
from PIL import Image, ImageOps

def process_images_in_folder(folder_path, target_size=(48, 48), inner_size=(32, 32)):
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не найдена.")
        return
    # Проходимся по всем файлам в папке
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Проверяем, является ли файл изображением
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Открываем изображение
                img = Image.open(file_path)
                img = img.convert("RGBA")

                # Узнаем размеры
                width, height = img.size

                # Если хотя бы одна сторона меньше inner_size, увеличиваем до inner_size с сохранением пропорций
                if width < inner_size[0] or height < inner_size[1]:
                    # Масштабируем вверх
                    img = ImageOps.contain(img, inner_size, method=Image.Resampling.LANCZOS)
                else:
                    # Уменьшаем при необходимости
                    img.thumbnail(inner_size, Image.Resampling.LANCZOS)

                # Создаём новое изображение target_size
                outer_img = Image.new('RGBA', target_size, (0, 0, 0, 255))

                # Центрируем изображение
                x_offset = (target_size[0] - img.size[0]) // 2
                y_offset = (target_size[1] - img.size[1]) // 2

                outer_img.paste(img, (x_offset, y_offset), mask=img if img.mode == "RGBA" else None)

                # Сохраняем результат
                outer_img.save(file_path)
                print(f"Изображение {filename} обработано и сохранено.")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

