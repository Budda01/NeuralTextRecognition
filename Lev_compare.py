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

if __name__ == "__main__":
    file1_path = "output.txt"
    file2_path = "RESULT.txt"

    text1 = read_file(file1_path)
    text2 = read_file(file2_path)

    score = normalized_levenshtein(text1, text2)
    similarity = 1 - score

    print(f"\nНормализованное расстояние Левенштейна: {score:.4f}")
    print(f"Сходство текстов: {similarity * 100:.2f}%")
