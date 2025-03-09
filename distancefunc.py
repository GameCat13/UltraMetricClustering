import numpy as np

def matrix_distance(matrix1, matrix2, norm_type=1):
    """
    Вычисляет расстояние между двумя матрицами с использованием нормы.
    :param matrix1: Первая матрица (numpy array).
    :param matrix2: Вторая матрица (numpy array).
    :param norm_type: Тип нормы ('fro' для Frobenius, 1 для L1, 2 для L2 и т.д.).
    :return: Расстояние между матрицами.
    """
    difference = matrix1 - matrix2
    return np.linalg.norm(difference, ord=norm_type)


def relative_error(matrix1, matrix2, norm='l1_norm', percent=True):
    """
    Вычисляет относительную ошибку между двумя матрицами расстояний, игнорируя диагональ.
    :param matrix1: Первая матрица расстояний (numpy array).
    :param matrix2: Вторая матрица расстояний (numpy array).
    :param norm: Норма для вычисления ошибки ('L1', 'L2', 'Linf').
    :return: Относительная ошибка вне диагонали.
    """
    # Маска для выделения внедиагональных элементов
    mask = np.eye(matrix1.shape[0]) == 0
    masked_matrix1 = matrix1[mask]
    masked_matrix2 = matrix2[mask]

    # Вычисление разности между матрицами
    diff = masked_matrix1 - masked_matrix2

    # Выбор нормы
    if norm == 'l1_norm':
        error = np.linalg.norm(diff, ord=1) / np.linalg.norm(masked_matrix1, ord=1)
    elif norm == 'l2_norm':
        error = np.linalg.norm(diff, ord=2) / np.linalg.norm(masked_matrix1, ord=2)
    elif norm == 'linf_norm':
        error = np.linalg.norm(diff, ord=np.inf) / np.linalg.norm(masked_matrix1, ord=np.inf)
    else:
        raise ValueError("Норма должна быть 'L1', 'L2' или 'Linf'.")

    return error * 100 if percent else error