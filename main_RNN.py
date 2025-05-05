import os
import numpy as np
import cv2
import re
import shutil
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from divider import image_divider
from pre_proc import process_images_in_folder

# === Параметры ===
IMG_H, IMG_W = 48, 48
MAX_WORD_LEN = 20
MODEL_PATH = "RNN/prediction_model.keras"
CHAR_IMAGE_DIR = "images"
RESULT_FILE = "RESULT_RNN.txt"

alphabet = list("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ" +
                "абвгдежзиклмнопрстуфхцчшщъьэюя" +
                ",.()Ыы")
idx_to_char = {i: ch for i, ch in enumerate(alphabet)}

# === Утилиты ===
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=True)
    results = decoded[0].numpy()
    texts = []
    for res in results:
        texts.append(''.join([idx_to_char.get(i, '') for i in res if i >= 0]))
    return texts

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.
    return np.expand_dims(img, axis=-1)

def pad_sequence(sequence):
    padded = list(sequence)
    while len(padded) < MAX_WORD_LEN:
        padded.append(np.zeros((IMG_H, IMG_W, 1), dtype=np.float32))
    return np.expand_dims(np.array(padded[:MAX_WORD_LEN]), axis=0)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

def write_results_to_file(results, file_name=RESULT_FILE):
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

# === Главная логика ===
if __name__ == "__main__":
    # Разделение и предобработка
    len_char = image_divider('test_image/img_5.png')
    print(f"Обработано символов: {len_char}")
    process_images_in_folder(CHAR_IMAGE_DIR)

    # Загрузка модели
    model = tf.keras.models.load_model(MODEL_PATH)

    results = []
    current_sequence = []
    dark_count = 0

    for i in range(len_char):
        img_path = os.path.join(CHAR_IMAGE_DIR, f"char_{i}.png")
        preprocessed = preprocess_image(img_path)
        if preprocessed is None:
            continue

        if np.sum(preprocessed) == 0:
            dark_count += 1
            if dark_count == 1 and current_sequence:
                padded = pad_sequence(current_sequence)
                prediction = model.predict(padded)
                text = decode_prediction(prediction)[0]
                results.append(text + " ")
                current_sequence = []
            elif dark_count == 2:
                results.append("\n")
                dark_count = 0
        else:
            dark_count = 0
            current_sequence.append(preprocessed)

    # Последнее слово
    if current_sequence:
        padded = pad_sequence(current_sequence)
        prediction = model.predict(padded)
        text = decode_prediction(prediction)[0]
        results.append(text)

    # Постобработка
    for i in range(len(results)):
        results[i] = results[i].replace('ьI', 'ы')
        results[i] = results[i].replace('ЬI', 'ы')
        results[i] = results[i].replace('Е..', 'Ё')
        results[i] = results[i].replace('е..', 'ё')

    write_results_to_file(results)
    clear_folder(CHAR_IMAGE_DIR)
    fix_text_file(RESULT_FILE)
    print(f"Результаты сохранены в {RESULT_FILE}")
