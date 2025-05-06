import cv2
import numpy as np
import os
from binarizer import improved_binarizer
import matplotlib.pyplot as plt
def flood_fill(binary, visited, x, y, fill_value=0):
    h, w = binary.shape
    stack = [(x, y)]
    pixels = []

    # Все 8 направлений (включая диагонали)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (1, -1), (-1, 1), (1, 1)]

    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        if visited[cy, cx] or binary[cy, cx] == fill_value:
            continue
        visited[cy, cx] = True
        pixels.append((cx, cy))
        for dx, dy in directions:
            stack.append((cx + dx, cy + dy))

    return pixels


def find_full_symbol(binary, visited, start_x, start_y):
    """
    Находит полный символ, начиная с указанной точки.
    """
    h, w = binary.shape
    if visited[start_y, start_x] or binary[start_y, start_x] == 0:
        return 0, 0, 0, 0

    # Находим все пиксели символа
    pixels = flood_fill(binary, visited, start_x, start_y)

    if not pixels:
        return 0, 0, 0, 0

    # Вычисляем границы символа
    x_coords = [p[0] for p in pixels]
    y_coords = [p[1] for p in pixels]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Обновляем посещённые точки
    visited[y_min:y_max + 1, x_min:x_max + 1] = True

    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

def find_lines(binary):
    """
    Определяет строки текста на бинаризованном изображении.
    """
    projection = np.sum(binary == 255, axis=1)
    mean_value = np.mean(projection)

    lines = []
    in_region = False
    start_y = 0
    for y, value in enumerate(projection):
        if value > 0.7 * mean_value and not in_region:
            in_region = True
            start_y = y
        elif value <= 0.3 * mean_value and in_region:
            in_region = False
            lines.append((start_y, y))
    if in_region:
        lines.append((start_y, binary.shape[0]))
    return lines

def extract_characters_from_line(binary, visited, line_start, line_end):
    line_chars = []
    for y in range(line_start, line_end):
        for x in range(binary.shape[1]):
            if visited[y, x] or binary[y, x] == 0:
                continue
            symbol_x, symbol_y, symbol_w, symbol_h = find_full_symbol(binary, visited, x, y)
            line_chars.append((symbol_x, symbol_y, symbol_w, symbol_h))
    line_chars.sort(key=lambda char: char[0])
    return line_chars

def calculate_average_distance(line_chars):
    if len(line_chars) > 1:
        distances = [line_chars[i + 1][0] - (line_chars[i][0] + line_chars[i][2]) for i in range(len(line_chars) - 1)]
        return sum(distances) / len(distances)
    return 0

def save_character(binary, char_index, symbol_x, symbol_y, symbol_w, symbol_h, output_folder, f):
    additional_top = symbol_h // 4
    extended_symbol_y = max(0, symbol_y - additional_top)
    extended_symbol_h = symbol_h + (symbol_y - extended_symbol_y)
    char_image = binary[extended_symbol_y:extended_symbol_y + extended_symbol_h, symbol_x:symbol_x + symbol_w]
    char_path = os.path.join(output_folder, f"char_{char_index}.png")
    cv2.imwrite(char_path, char_image)
    f.write(f"  Символ {char_index}: {char_path}\n")
    return char_index + 1

def save_black_image(char_index, black_image, output_folder, f):
    black_image_path = os.path.join(output_folder, f"char_{char_index}.png")
    cv2.imwrite(black_image_path, black_image)
    f.write(f"  Черное изображение (разрыв): {black_image_path}\n")
    return char_index + 1

def process_lines(binary, visited, lines, output_folder, output_text_file):
    detected_chars = []
    char_index = 0
    black_image = np.zeros((binary.shape[0] // 10, binary.shape[1] // 10), dtype=np.uint8)
    black_image_count = 0

    with open(output_text_file, 'w', encoding='utf-8') as f:
        for line_num, (line_start, line_end) in enumerate(lines, start=1):
            f.write(f"Строка {line_num}:\n")

            # Извлечение символов из строки
            line_chars = extract_characters_from_line(binary, visited, line_start, line_end)
            # Рассчитываем среднее расстояние между символами
            avg_distance = calculate_average_distance(line_chars)
            # Сохраняем символы
            for i, (symbol_x, symbol_y, symbol_w, symbol_h) in enumerate(line_chars):
                char_index = save_character(binary, char_index, symbol_x, symbol_y, symbol_w, symbol_h, output_folder, f)

                # Если расстояние до следующего символа больше 1.5 среднего расстояния, добавляем чёрное изображение
                if i < len(line_chars) - 1:
                    next_x = line_chars[i + 1][0]
                    gap = next_x - (symbol_x + symbol_w)
                    if gap > 1.5 * avg_distance:
                        char_index = save_black_image(char_index, black_image, output_folder, f)
                        black_image_count += 1

                # Добавляем символ в список обнаруженных символов
                detected_chars.append((symbol_x, symbol_y, symbol_w, symbol_h))

            # Добавляем два чёрных изображения в конце строки
            char_index = save_black_image(char_index, black_image, output_folder, f)
            black_image_count += 1
            char_index = save_black_image(char_index, black_image, output_folder, f)
            black_image_count += 1

    return detected_chars, black_image_count

def image_divider(file_path):
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)
    output_text_file = "output_symbols.txt"
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    binary = improved_binarizer(image)
    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    lines = find_lines(binary)
    detected_chars, black_image_count = process_lines(binary, visited, lines, output_folder, output_text_file)

    # Рисуем прямоугольники вокруг обнаруженных символов
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in detected_chars:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('Detected Characters', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(detected_chars) + black_image_count