import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense,
                                     Bidirectional, LSTM, TimeDistributed, Flatten, Lambda, Masking)
import tensorflow.keras.backend as K

# === Параметры ===
IMG_H, IMG_W = 48, 48
MAX_WORD_LEN = 20
BATCH_SIZE = 16
EPOCHS = 17
VARIANTS_PER_WORD = 30
TIME_STEPS = MAX_WORD_LEN

WORD_FILE = "RUS2.txt"
LETTER_DIR = "../DataSet/output_imag"

alphabet = list("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ" +
                "абвгдежзиклмнопрстуфхцчшщъьэюя" +
                ",.()Ыы")
char_to_idx = {char: i for i, char in enumerate(alphabet)}
idx_to_char = {i: char for i, char in enumerate(alphabet)}

def load_words(word_file, variants_per_word):
    with open(word_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return [word for word in words for _ in range(variants_per_word)]

def get_folder_index(ch):
    if ch in "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ":
        return 1 + "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЬЭЮЯ".index(ch)
    elif ch in "абвгдежзиклмнопрстуфхцчшщъьэюя":
        return 31 + "абвгдежзиклмнопрстуфхцчшщъьэюя".index(ch)
    return {',': 61, '.': 62, '(': 63, ')': 64, 'I': 65}[ch]

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

def simple_generator(words, letter_dir, batch_size=16):
    while True:
        random.shuffle(words)
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            X_data = np.zeros((len(batch_words), MAX_WORD_LEN, IMG_H, IMG_W, 1), dtype=np.float32)
            labels = np.ones([len(batch_words), MAX_WORD_LEN]) * -1
            input_length = np.zeros((len(batch_words), 1), dtype=np.int32)
            label_length = np.zeros((len(batch_words), 1), dtype=np.int32)

            for j, word in enumerate(batch_words):
                letter_imgs = []
                letter_ids = []

                for ch in word:
                    if ch == 'Й': ch = 'И'
                    elif ch == 'й': ch = 'и'
                    if ch not in char_to_idx:
                        continue
                    img_set = char_to_image(ch, letter_dir)
                    letter_imgs.extend(img_set)
                    letter_ids.append(char_to_idx[ch])

                letter_imgs = letter_imgs[:MAX_WORD_LEN]
                letter_ids = letter_ids[:MAX_WORD_LEN]

                for k in range(len(letter_imgs)):
                    X_data[j, k] = letter_imgs[k]
                for k in range(len(letter_ids)):
                    labels[j, k] = letter_ids[k]

                input_length[j] = TIME_STEPS
                label_length[j] = len(letter_ids)

            inputs = {
                "image": X_data,
                "label": labels.astype(np.int32),
                "input_length": input_length,
                "label_length": label_length
            }
            outputs = np.zeros([len(batch_words)])
            yield inputs, outputs

def build_model():
    input_img = Input(shape=(MAX_WORD_LEN, IMG_H, IMG_W, 1), name="image")
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(input_img)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = Masking(mask_value=0.0)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    y_pred = Dense(len(char_to_idx) + 1, activation='softmax', name="y_pred")(x)

    labels = Input(name='label', shape=[MAX_WORD_LEN], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    def ctc_lambda(args):
        y_pred, labels, input_len, label_len = args
        return K.ctc_batch_cost(labels, y_pred, input_len, label_len)

    loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    train_model = Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    prediction_model = Model(inputs=input_img, outputs=y_pred)

    return train_model, prediction_model

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=10, top_paths=1)
    results = decoded[0].numpy()
    texts = []
    for res in results:
        texts.append(''.join([idx_to_char.get(x, '') for x in res if x >= 0]))
    return texts

def train():
    words = load_words(WORD_FILE, VARIANTS_PER_WORD)
    steps_per_epoch = len(words) // BATCH_SIZE
    train_gen = simple_generator(words, LETTER_DIR, BATCH_SIZE)

    model, prediction_model = build_model()

    # ВАЖНО: правильно указываем имя выхода 'ctc'
    model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)
    model.summary()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=1)

        inputs, _ = next(train_gen)
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

    prediction_model.save("prediction_model.keras")
    print("Модель сохранена как 'prediction_model.keras'")

if __name__ == "__main__":
    train()
