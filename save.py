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


import pickle

def save_experiment(results, experiment_dir, experiment_name):
    """
    Сохраняет результаты эксперимента в файл.

    Параметры:
    -----------
    results : dict
        Словарь с результатами эксперимента.
    experiment_dir : str
        Папка для сохранения.
    experiment_name : str
        Имя файла.
    """
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    file_path = os.path.join(experiment_dir, f"{experiment_name}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)