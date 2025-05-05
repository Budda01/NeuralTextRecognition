import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow.keras.backend as K

# === Параметры ===
IMG_H, IMG_W = 48, 48
MAX_WORD_LEN = 20
MODEL_PATH = "prediction_model.keras"

alphabet = list("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ" +
                "абвгдежзиклмнопрстуфхцчшщъьэюя" +
                ",.()Ыы")
idx_to_char = {i: ch for i, ch in enumerate(alphabet)}


def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=True)
    results = decoded[0].numpy()
    texts = []
    for res in results:
        texts.append(''.join([idx_to_char.get(i, '') for i in res if i >= 0]))
    return texts


def load_sequence_from_folder(folder_path):
    files = sorted([f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.png', '.jpg'))],
                   key=lambda x: int(os.path.splitext(x)[0]))

    letter_imgs = []
    for fname in files[:MAX_WORD_LEN]:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Ошибка загрузки изображения: {fname}")
        img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.
        letter_imgs.append(np.expand_dims(img, axis=-1))

    # Паддинг нулями
    while len(letter_imgs) < MAX_WORD_LEN:
        letter_imgs.append(np.zeros((IMG_H, IMG_W, 1), dtype=np.float32))

    return np.expand_dims(np.array(letter_imgs), axis=0)  # (1, MAX_WORD_LEN, 48, 48, 1)


def predict_from_folder(folder_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    input_data = load_sequence_from_folder(folder_path)
    pred = model.predict(input_data)
    text = decode_prediction(pred)[0]
    print(f" Предсказанное слово из '{folder_path}': {text}")


# === Пример
if __name__ == "__main__":
    predict_from_folder("../DataSet/WordGen/sample")
