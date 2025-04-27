import os
import shutil
from divider import image_divider
from pre_proc import process_images_in_folder


def clear_folder(folder_path):
    """Удаляет все файлы и папки в указанной директории."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")


def get_max_image_index(folder_path):
    """Определяет максимальный индекс изображений в папке."""
    max_index = 0
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png") and file_name[:-4].isdigit():
                file_index = int(file_name[:-4])
                max_index = max(max_index, file_index)
    return max_index


def main():
    input_folder = "output_images"
    temp_folder = "images"

    for i in range(1, 61):
        input_file = os.path.join(input_folder, f"{i}.png")
        output_subfolder = os.path.join("output_imag", str(i))  # от 31 до 60

        if not os.path.exists(input_file):
            print(f"Изображение {input_file} не найдено. Пропуск...")
            continue

        print(f"Обрабатываем изображение {input_file}...")

        # Создаём подкаталог под текущий результат
        os.makedirs(output_subfolder, exist_ok=True)

        # Вызываем image_divider для обработки изображения
        image_divider(input_file)

        # Обрабатываем изображения в temp_folder
        process_images_in_folder(temp_folder)

        # Определяем текущий максимальный индекс в папке (на случай, если уже есть файлы)
        max_image_index = get_max_image_index(output_subfolder)

        # Копируем содержимое temp_folder в подкаталог
        for file_name in os.listdir(temp_folder):
            src_path = os.path.join(temp_folder, file_name)
            max_image_index += 1
            new_file_name = f"{max_image_index}.png"
            dest_path = os.path.join(output_subfolder, new_file_name)
            shutil.move(src_path, dest_path)

        print(f"Файлы из {temp_folder} перемещены в {output_subfolder}.")

        # Очищаем временную папку
        clear_folder(temp_folder)

    print("Обработка завершена.")


if __name__ == "__main__":
    main()
