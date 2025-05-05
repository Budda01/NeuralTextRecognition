import tensorflow as tf

print("==== TensorFlow GPU Check ====")

# Проверка доступных физических устройств GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Найдено {len(gpus)} GPU:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("❌ GPU не найден. Вычисления будут идти на CPU.")

# Включаем логирование размещения операций
tf.debugging.set_log_device_placement(True)

# Небольшая тестовая модель для проверки устройства
try:
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("✅ Операция матричного умножения завершена.")
except RuntimeError as e:
    print("❌ Ошибка при выполнении операции:", e)
