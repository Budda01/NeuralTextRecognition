import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time
import os

# Начало таймера
t1 = time.monotonic()

# Функция для преобразования изображения
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Ошибка: файл {image_path} не загружен.")
            return None
        resized_image = cv2.resize(image, (48, 48))
        return resized_image
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return None

# Папка, содержащая изображения
data_folder = 'DataSet/output_imag'

# Определяем количество классов по количеству папок
folders = sorted([folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))])
num_classes = len(folders)
print(folders)
# Массивы для хранения изображений и меток
images = []
labels = []

# Чтение изображений из каждой папки
for folder_index, folder_name in enumerate(folders):
    folder_path = os.path.join(data_folder, folder_name)
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is not None:
            images.append(preprocessed_image)
            labels.append(folder_index)

# Преобразование в массивы numpy
images = np.array(images)
labels = np.array(labels)

# Разделение данных на обучающий и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=33)

# Подготовка данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Проверка существования сохранённой модели
model_path = "alphabet_recognition_model.h5"

if os.path.exists(model_path):
    print("Загружаем сохранённую модель...")
    model = load_model(model_path)
else:
    print("Обучаем новую модель...")

    # Улучшенная структура модели
    image_input = Input(shape=(48, 48, 1))
    conv1 = Conv2D(48, (3, 3), padding='same', activation='relu')(image_input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flatten = Flatten()(pool3)
    dense1 = Dense(512, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    output = Dense(num_classes, activation='softmax')(dropout1)

    model = Model(inputs=image_input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    history = model.fit(x_train, y_train, epochs=12, batch_size=64, validation_data=(x_test, y_test))
    # Сохранение модели
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc = history_dict['accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Потери на этапе обучения')
plt.plot(epochs, val_loss_values, 'b', label='Потери на этапе проверки')
plt.title('Потери на этапе обучения и проверки')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Точность на этапе обучения')
plt.plot(epochs, val_acc_values, 'b', label='Точность на этапе проверки')
plt.title('Точность на этапах обучения и проверки')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show
# Оценка модели
loss, accuracy = model.evaluate(x_test, y_test)
print("Точность на тестовых данных:", accuracy)

# Построение графиков
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

plt.figure(figsize=(14, 6))
for i in range(40):
    plt.subplot(4, 10, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap=plt.cm.binary)
    plt.title(f"Actual: {folders[y_test[i]]}\nPred: {folders[pred[i]]}", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Графики неверных предсказаний
mask = pred != y_test
x_false = x_test[mask]
y_false = y_test[mask]
y_pred_false = pred[mask]

plt.figure(figsize=(14, 6))
for i in range(min(25, len(x_false))):  # Лимит на 25 неверных предсказаний
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_false[i].squeeze(), cmap=plt.cm.binary)
    plt.title(f"Actual: {folders[y_false[i]]}\nPred: {folders[y_pred_false[i]]}", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Папка с тестовыми изображениями
custom_test_folder = "images"

# Преобразование изображений из папки
custom_images = []
custom_image_paths = []
for filename in os.listdir(custom_test_folder):
    image_path = os.path.join(custom_test_folder, filename)
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        custom_images.append(preprocessed_image)
        custom_image_paths.append(filename)

# Если изображения найдены
if custom_images:
    custom_images = np.array(custom_images)
    custom_images = custom_images.astype('float32') / 255.0
    custom_images = np.expand_dims(custom_images, axis=-1)

    # Предсказания для пользовательских изображений
    custom_predictions = model.predict(custom_images)
    custom_pred_classes = np.argmax(custom_predictions, axis=1)

    # Вывод результатов
    plt.figure(figsize=(14, 6))
    for i in range(len(custom_images)):
        plt.subplot(4, 10, i + 1)
        plt.imshow(custom_images[i].squeeze(), cmap=plt.cm.binary)
        plt.title(f"Pred: {folders[custom_pred_classes[i]]}\n{custom_image_paths[i]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Нет изображений для тестирования в папке", custom_test_folder)
