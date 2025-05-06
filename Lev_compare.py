import Levenshtein

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Файл не найден: {filepath}")
        return ""

def normalized_levenshtein(text1, text2):
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 0.0
    return distance / max_len

def find_first_difference(text1, text2):
    min_len = min(len(text1), len(text2))
    for i in range(min_len):
        if text1[i] != text2[i]:
            return i, text1[i], text2[i]
    if len(text1) != len(text2):
        return min_len, text1[min_len:min_len+1], text2[min_len:min_len+1]
    return -1, '', ''

def print_difference_info(text1, text2, label):
    index, char1, char2 = find_first_difference(text1, text2)
    if index == -1:
        print(f"\n{label}: Тексты идентичны.")
    else:
        print(f"\n{label}: Первое различие на позиции {index}:")
        print(f" → output.txt      : '{char1}'")
        print(f" → {label}.txt     : '{char2}'")
        context_start = max(0, index - 10)
        context_end = index + 10
        print(f"\nКонтекст вокруг различия:")
        print(f" → output.txt      : '{text1[context_start:context_end]}'")
        print(f" → {label}.txt     : '{text2[context_start:context_end]}'")

if __name__ == "__main__":
    file1_path = "output.txt"
    file2_path = "RESULT_RNN.txt"
    file3_path = "RESULT_CNN.txt"

    text1 = read_file(file1_path)
    text2 = read_file(file2_path)
    text3 = read_file(file3_path)

    score1 = normalized_levenshtein(text1, text2)
    score2 = normalized_levenshtein(text1, text3)

    similarity1 = 1 - score1
    similarity2 = 1 - score2

    print(f"\nНормализованное расстояние Левенштейна RNN: {score1:.4f}")
    print(f"Сходство текстов: {similarity1 * 100:.2f}%")

    print(f"\nНормализованное расстояние Левенштейна CNN: {score2:.4f}")
    print(f"Сходство текстов: {similarity2 * 100:.2f}%")

    print_difference_info(text1, text2, "RESULT_RNN")
    print_difference_info(text1, text3, "RESULT_CNN")
