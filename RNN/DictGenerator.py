import os
import cv2
import numpy as np
import random

# Пути
BASE_DIR = "../DataSet/output_imag"
WORDS_FILE = "popular.txt"
OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Словарь соответствия символов папкам
char_to_folder = {}

uppercase_letters = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ"
for i, ch in enumerate(uppercase_letters, start=1):
    char_to_folder[ch] = str(i)

lowercase_letters = uppercase_letters.lower()
for i, ch in enumerate(lowercase_letters, start=31):
    char_to_folder[ch] = str(i)

char_to_folder.update({
    ',': "61",
    '.': "62",
    '(': "63",
    ')': "64",
    'I': "65",
})


def get_random_image_from_folder(folder_path):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise Exception(f"Нет изображений в папке {folder_path}")
    random_file = random.choice(images)
    img_path = os.path.join(folder_path, random_file)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return image


def build_word_image(word: str) -> np.ndarray:
    letter_images = []
    for ch in word:
        if ch == 'Ы':
            components = ['Ь', 'I']
        elif ch == 'ы':
            components = ['ь', 'I']
        elif ch == 'й':
            components = ['и']
        elif ch == 'Й':
            components = ['И']
        else:
            components = [ch]

        for sub_ch in components:
            folder_number = char_to_folder.get(sub_ch)
            if not folder_number:
                raise ValueError(f"Символ '{sub_ch}' не поддерживается.")
            folder_path = os.path.join(BASE_DIR, folder_number)
            letter_image = get_random_image_from_folder(folder_path)
            if letter_image is None:
                raise Exception(f"Не удалось загрузить символ '{sub_ch}' из {folder_path}")
            desired_height = 48
            h, w = letter_image.shape
            if h != desired_height:
                scale = desired_height / h
                letter_image = cv2.resize(letter_image, (int(w * scale), desired_height))
            letter_images.append(letter_image)
    return np.hstack(letter_images)


# Загружаем список слов
with open(WORDS_FILE, "r", encoding="utf-8") as f:
    word_list = [line.strip() for line in f if line.strip()]

# Сохраняем соответствие папка — слово
words_txt_path = os.path.join(OUTPUT_DIR, "words.txt")
with open(words_txt_path, "w", encoding="utf-8") as wf:

    # Генерация слов
    for index, word in enumerate(word_list, start=1):
        folder_name = str(index)
        word_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(word_folder, exist_ok=True)

        # Узнаем текущий max номер файла в папке
        existing_files = [f for f in os.listdir(word_folder) if f.endswith('.png')]
        existing_indices = [
            int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith(folder_name + "_")
        ]
        next_index = max(existing_indices, default=-1) + 1

        # Генерируем 5 вариантов изображения слова
        for i in range(5):
            try:
                image = build_word_image(word)
                filename = f"{folder_name}_{next_index + i}.png"
                filepath = os.path.join(word_folder, filename)
                cv2.imwrite(filepath, image)
            except Exception as e:
                print(f"Ошибка генерации для слова '{word}': {e}")

        # Запись в words.txt
        wf.write(f"{folder_name} {word}\n")

print("✅ Генерация завершена. Слова сохранены, файл words.txt создан.")
