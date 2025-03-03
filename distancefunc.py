import numpy as np

def matrix_distance(matrix1, matrix2, norm_type='fro'):
    """
    Вычисляет расстояние между двумя матрицами с использованием нормы.
    :param matrix1: Первая матрица (numpy array).
    :param matrix2: Вторая матрица (numpy array).
    :param norm_type: Тип нормы ('fro' для Frobenius, 1 для L1, 2 для L2 и т.д.).
    :return: Расстояние между матрицами.
    """
    difference = matrix1 - matrix2
    return np.linalg.norm(difference, ord=norm_type)


def relative_error(matrix1, matrix2):
    """
    Вычисляет среднюю относительную ошибку между двумя матрицами расстояний, игнорируя диагональ.
    :param matrix1: Первая матрица расстояний (numpy array).
    :param matrix2: Вторая матрица расстояний (numpy array).
    :return: Средняя относительная ошибка вне диагонали.
    """
    mask = np.eye(matrix1.shape[0]) == 0 # Маска для выделения внедиагональных элементов
    masked_matrix1 = matrix1[mask]
    masked_matrix2 = matrix2[mask]
    return np.mean(np.abs((masked_matrix1 - masked_matrix2) / masked_matrix1))