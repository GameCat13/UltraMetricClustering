import numpy as np
from scipy.spatial.distance import pdist, squareform

# Генерация случайных строк
def generate_random_strings(num_strings, length, alphabet):
    return [''.join(np.random.choice(list(alphabet), length)) for _ in range(num_strings)]

# Функция для вычисления расстояния Хэмминга между двумя строками
def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Параметры
num_strings = 5  # Количество строк
length = 20      # Длина строк
alphabet = 'ACGT'  # Алфавит из 4 символов

# Генерация строк
strings = generate_random_strings(num_strings, length, alphabet)
print("Сгенерированные строки:")
for s in strings:
    print(s)

# Вычисление попарных расстояний Хэмминга
distances = np.zeros((num_strings, num_strings))
for i in range(num_strings):
    for j in range(i + 1, num_strings):
        distances[i, j] = hamming_distance(strings[i], strings[j])
        distances[j, i] = distances[i, j]  # Расстояние симметрично

print("\nМатрица попарных расстояний Хэмминга:")
print(distances)