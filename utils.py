import numpy as np

# Проверка ультраметрической матрицы расстояний
def is_ultrametric(matrix):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):  # Проверяем только верхний треугольник
            for k in range(n):
                if k == i or k == j:
                    continue  # Пропускаем случаи, когда k совпадает с i или j
                if matrix[i, j] > max(matrix[i, k], matrix[k, j]):
                    return False
    return True
