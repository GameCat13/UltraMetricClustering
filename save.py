import numpy as np
import os

# Сохранение матрицы в файл
def save_matrix(matrix):
    """
    Сохраняет матрицу в файл.

    Параметры:
    ----------
    matrix : ndarray
        Матрица для сохранения.
    """
    n = matrix.shape[0]
    t = 1
    while True:
        filename = f"matrix({n}x{n})_{t}.txt"
        if not os.path.exists(filename):
            break
        t += 1
    np.savetxt(filename, matrix, fmt="%.6f")
    print(f"Матрица сохранена в файл: {filename}")