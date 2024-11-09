import pandas as pd
import os

# Шлях до вашого великого файлу CSV
input_file = '../../data_set/unbalaced_20_80_dataset.csv'
# Ім'я папки для збереження частин
output_folder = 'DATASET'
# Розмір блоку для зчитування
chunksize = 10**5

# Створюємо папку DATASET, якщо вона ще не існує
os.makedirs(output_folder, exist_ok=True)

# Лічильник для іменування нових файлів
file_counter = 1

try:
    # Читаємо файл частинами і зберігаємо кожен блок у новий файл у папці DATASET
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        output_file = os.path.join(output_folder, f'chunk_{file_counter}.csv')
        chunk.to_csv(output_file, index=False)  # Зберігає chunk у новий файл без індекса
        print(f'Створено файл: {output_file}')
        file_counter += 1
except Exception as e:
    print("Помилка під час обробки:", e)
