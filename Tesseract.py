from PIL import Image
import pytesseract
import cv2
import os

# Пути
image_path = 'test_image/img_12.png'
temp_path = 'temp_processed.png'

# Обработка изображения
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite(temp_path, gray)

# Установка путей к tesseract и tessdata
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Распознавание текста
text = pytesseract.image_to_string(Image.open(temp_path), lang='rus')

# Удаляем временный файл
os.remove(temp_path)

# Сохраняем в txt
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Распознанный текст сохранён в output.txt")
