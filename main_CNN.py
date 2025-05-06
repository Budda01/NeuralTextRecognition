from divider import image_divider
from pre_proc import process_images_in_folder
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import re

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

def preprocess_image(image_path):
    try:
        # Загружаем изображение в градациях серого
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Ошибка: файл {image_path} не загружен.")
            return None
        # Изменение размера изображения
        resized_image = cv2.resize(image, (48, 48))
        return resized_image
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return None

def predict_single_image(model, image, folders):
    # Преобразование изображения
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Предсказание
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return folders[predicted_class]

def write_results_to_file(results, file_name="RESULT_CNN.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        file.writelines(results)

def fix_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return

    upper_count = sum(1 for c in letters if c.isupper())
    lower_count = sum(1 for c in letters if c.islower())

    if upper_count > lower_count:
        corrected_text = text.upper()
    else:
        text = text.lower()
        sentences = re.split(r'([.!?]["\']?\s*)', text)
        corrected = ''
        capitalize_next = True

        for part in sentences:
            if capitalize_next:
                corrected += part.capitalize()
            else:
                corrected += part
            capitalize_next = bool(re.match(r'[.!?]["\']?\s*$', part))
        corrected_text = corrected
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(corrected_text)


if __name__ == "__main__":
    model_path = "CNN/alphabet_recognition_model.h5"
    len_char = image_divider('test_image/img_23.png')
    custom_test_folder = "images"
    print(f"Обработано символов: {len_char}")
    process_images_in_folder(custom_test_folder)
    fol = ['А','Б','В','Г','Д','Е','Ж','З','И','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ь','Э','Ю','Я','а','б','в','г','д','е','ж','з','и','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ', 'ъ', 'ь', 'э', 'ю', 'я', ',', '.', '(', ')', 'I']
    folders = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '7', '8', '9']
    fol3 = [fol[int(folder) - 1] for folder in folders]

    # Загрузка модели
    model = load_model(model_path)

    results = []
    current_line = ""
    dark_count = 0

    for i in range(len_char):
        test_image_path = os.path.join(custom_test_folder, f"char_{i}.png")
        preprocessed_image = preprocess_image(test_image_path)
        pixel_sum = np.sum(preprocessed_image)

        if pixel_sum == 0:
            dark_count += 1
            if dark_count == 1:
                current_line += " "
            elif dark_count == 2:
                results.append(current_line.strip() + "\n")
                current_line = ""
                dark_count = 0
        else:
            dark_count = 0
            if preprocessed_image is not None:
                predicted_char = predict_single_image(model, preprocessed_image, fol3)
                current_line += predicted_char

    # Добавление последней строки, если она не пуста
    if current_line.strip():
        results.append(current_line.strip() + "\n")
    for i in range(len(results)):
        results[i] = results[i].replace('ьI', 'ы')
        results[i] = results[i].replace('ЬI', 'ы')
        results[i] = results[i].replace('Е..', 'Ё')
        results[i] = results[i].replace('е..', 'ё')


    write_results_to_file(results)
    clear_folder(custom_test_folder)
    # fix_text_file("RESULT_CNN.txt")
