import matplotlib.pyplot as plt

def visualize_matrix(matrix):
    """
    Визуализирует матрицу в виде тепловой карты.

    Параметры:
    ----------
    matrix : ndarray
        Матрица для визуализации.
    """
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Тепловая карта матрицы")
    plt.show()
