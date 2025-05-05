import os
import cv2
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense,
                                     Bidirectional, LSTM, TimeDistributed, Flatten, Lambda)
import tensorflow.keras.backend as K

# === Параметры ===
IMG_H, IMG_W = 48, 48
MAX_WORD_LEN = 20
BATCH_SIZE = 16
EPOCHS = 8
VARIANTS_PER_WORD = 25

WORD_FILE = "RUS2.txt"
LETTER_DIR = "../DataSet/output_imag"

alphabet = list("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ" +
                "абвгдежзиклмнопрстуфхцчшщъьэюя" +
                ",.()Ыы")
char_to_idx = {char: i for i, char in enumerate(alphabet)}
idx_to_char = {i: char for i, char in enumerate(alphabet)}

# === Утилиты ===
def load_words(word_file, variants_per_word):
    with open(word_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return [word for word in words for _ in range(variants_per_word)]

def get_folder_index(ch):
    if ch in "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ":
        return 1 + "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ".index(ch)
    elif ch in "абвгдежзиклмнопрстуфхцчшщъьэюя":
        return 31 + "абвгдежзиклмнопрстуфхцчшщъьэюя".index(ch)
    return {
        ',': 61,
        '.': 62,
        '(': 63,
        ')': 64,
        'I': 65
    }[ch]

def char_to_image(ch, letter_dir):
    if ch == 'Ы':
        chars = ['Ь', 'I']
    elif ch == 'ы':
        chars = ['ь', 'I']
    else:
        chars = [ch]

    imgs = []
    for sub_ch in chars:
        folder_idx = get_folder_index(sub_ch)
        folder = os.path.join(letter_dir, str(folder_idx))
        files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))]
        if not files:
            raise ValueError(f"Нет изображений для {sub_ch} в {folder}")
        file = random.choice(files)
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.
        imgs.append(np.expand_dims(img, axis=-1))
    return imgs

# === Генератор ===
def simple_generator(words, letter_dir, batch_size=16):
    while True:
        random.shuffle(words)
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            X_data = np.zeros((len(batch_words), MAX_WORD_LEN, IMG_H, IMG_W, 1), dtype=np.float32)
            labels = np.ones([len(batch_words), MAX_WORD_LEN]) * -1
            input_length = np.zeros((len(batch_words), 1))
            label_length = np.zeros((len(batch_words), 1))

            for j, word in enumerate(batch_words):
                letter_imgs = []
                letter_ids = []

                for ch in word:
                    # Замены Й -> И, ё -> е (если нужно)
                    if ch == 'Й':
                        ch = 'И'
                    elif ch == 'й':
                        ch = 'и'
                    if ch not in char_to_idx:
                        continue  # Пропускаем неизвестные символы

                    img_set = char_to_image(ch, letter_dir)

                    # Добавляем все изображения в последовательность
                    letter_imgs.extend(img_set)

                    # Добавляем только один label на весь img_set
                    letter_ids.append(char_to_idx[ch])

                # Усечение, если длиннее MAX_WORD_LEN
                if len(letter_imgs) > MAX_WORD_LEN:
                    letter_imgs = letter_imgs[:MAX_WORD_LEN]

                if len(letter_ids) > MAX_WORD_LEN:
                    letter_ids = letter_ids[:MAX_WORD_LEN]

                for k in range(len(letter_imgs)):
                    X_data[j, k] = letter_imgs[k]

                for k in range(len(letter_ids)):
                    labels[j, k] = letter_ids[k]

                input_length[j] = MAX_WORD_LEN
                label_length[j] = len(letter_ids)

            inputs = {
                "image": X_data,
                "label": labels.astype(np.int32),
                "input_length": input_length,
                "label_length": label_length
            }
            outputs = {"ctc": np.zeros([len(batch_words)])}
            yield inputs, outputs

# === Модель ===
def build_model():
    input_img = Input(shape=(MAX_WORD_LEN, IMG_H, IMG_W, 1), name="image")
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(input_img)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    y_pred = Dense(len(char_to_idx) + 1, activation='softmax', name="y_pred")(x)

    labels = Input(name='label', shape=[MAX_WORD_LEN], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    def ctc_lambda(args):
        y_pred, labels, input_len, label_len = args
        tf.debugging.assert_less_equal(label_len, input_len, message="label_len > input_len!")
        return K.ctc_batch_cost(labels, y_pred, input_len, label_len)

    loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    return model


def build_prediction_model(ctc_model):
    for input_tensor in ctc_model.inputs:
        if input_tensor.name.startswith("image"):
            input_img = input_tensor
            break
    else:
        raise ValueError("Не найден вход с именем, начинающимся на 'image'")

    y_pred = ctc_model.get_layer("y_pred").output
    return Model(inputs=input_img, outputs=y_pred)
# === Декодирование ===
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0].numpy()
    texts = []
    for res in results:
        texts.append(''.join([idx_to_char.get(x, '') for x in res if x >= 0]))
    return texts

# === Визуализация ===
def show_predictions(pred_model, inputs, count=8):
    preds = pred_model.predict(inputs["image"])
    pred_texts = decode_prediction(preds)

    true_texts = []
    for i in range(count):
        label = inputs["label"][i]
        true_texts.append(''.join([idx_to_char.get(int(x), '') for x in label if x != -1]))

    for i in range(count):
        letters = inputs["image"][i]
        plt.figure(figsize=(len(letters) * 1.5, 2))
        for j in range(MAX_WORD_LEN):
            img = letters[j].squeeze()
            if np.sum(img) == 0:
                continue
            plt.subplot(1, MAX_WORD_LEN, j + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.suptitle(f"True: {true_texts[i]}  |  Pred: {pred_texts[i]}")
        plt.show()

# === Обучение ===
def train():
    # Загрузка слов и генератор
    words = load_words(WORD_FILE, VARIANTS_PER_WORD)
    steps_per_epoch = len(words) // BATCH_SIZE
    train_gen = simple_generator(words, LETTER_DIR, BATCH_SIZE)

    # Построение и компиляция модели
    model = build_model()
    model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})
    model.summary()

    # Построение предсказательной модели до начала обучения
    prediction_model = build_prediction_model(model)

    # Обучение по эпохам
    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch + 1}/{EPOCHS}")
        model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=1)

        # Accuracy на одном батче после эпохи
        inputs, _ = next(simple_generator(words, LETTER_DIR, BATCH_SIZE))
        preds = prediction_model.predict(inputs["image"])
        decoded = decode_prediction(preds)

        true_texts = []
        for i in range(len(decoded)):
            label = inputs["label"][i]
            true_text = ''.join([idx_to_char.get(int(x), '') for x in label if x != -1])
            true_texts.append(true_text)

        correct = sum(t == p for t, p in zip(true_texts, decoded))
        acc = correct / len(decoded)
        print(f"Accuracy after epoch {epoch + 1}: {acc:.2%}")

    # Сохранение моделей
    model.save("ctc_model.keras")
    prediction_model.save("prediction_model.keras")
    print("Модели сохранены: 'ctc_model.keras' и 'prediction_model.keras'")




if __name__ == "__main__":
    train()
