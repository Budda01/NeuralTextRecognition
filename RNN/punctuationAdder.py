import random
import re

# Пути к файлам
input_filename = "RUS1.txt"
output_filename = "RUS2.txt"

# Разрешённые символы: А–Я, а–я, Ё, ё
allowed_chars = re.compile(r'^[А-Яа-яЁё]+$')

# Разрешённые знаки и куда вставлять
end_symbols = [",", ".", ")"]
start_symbols = ["("]

# Вероятность добавления знака
add_probability = 0.5

# Чтение, разбиение на слова и предобработка
with open(input_filename, "r", encoding="utf-8") as infile:
    raw_lines = infile.readlines()

raw_words = []
for line in raw_lines:
    # Разбиваем строку по пробелам
    for word in line.strip().split():
        # Заменяем буквы
        word = word.replace('ё', 'е').replace('Ё', 'Е')
        word = word.replace('й', 'и').replace('Й', 'И')
        # Проверка по шаблону
        if allowed_chars.fullmatch(word):
            raw_words.append(word)

# Применение знаков препинания
augmented_words = []
for word in raw_words:
    if random.random() < add_probability:
        symbol = random.choice(start_symbols + end_symbols)
        if symbol in end_symbols:
            word = word + symbol
        else:
            word = symbol + word
    augmented_words.append(word)

# Запись в файл: по одному слову на строку
with open(output_filename, "w", encoding="utf-8") as outfile:
    for word in augmented_words:
        outfile.write(word + "\n")

print(f"Готово! Обработано {len(raw_words)} слов. Сохранено в файл: {output_filename}")
