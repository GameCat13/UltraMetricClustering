import numpy as np

# Матрицы
matrix1 = np.array([
    [0, 2, 5],
    [2, 0, 3],
    [5, 3, 0]
])

matrix2 = np.array([
    [0, 1, 4],
    [1, 0, 2],
    [4, 2, 0]
])

# Разница матриц
difference = matrix1 - matrix2

# Верхний треугольник (без диагонали)
upper_triangle = np.triu(difference, k=1)

# L1-норма (сумма абсолютных значений элементов верхнего треугольника)
l1_norm = np.sum(np.abs(upper_triangle))

# L2-норма (корень из суммы квадратов элементов верхнего треугольника)
l2_norm = np.sqrt(np.sum(upper_triangle**2))

# Бесконечная норма (максимальное абсолютное значение среди элементов верхнего треугольника)
inf_norm = np.max(np.abs(upper_triangle))

print(f"L1-норма: {l1_norm}")
print(f"L2-норма: {l2_norm}")
print(f"Бесконечная норма: {inf_norm}")