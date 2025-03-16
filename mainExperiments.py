from input import *
from clustering import *
from utils import *
from distancefunc import *
from save import *
import numpy as np
from scipy.cluster.hierarchy import linkage
import pandas as pd

def run_experiment(n_points, n_dimensions=2, metric='euclidean', integer_values=False, random_seed=None):
    """
    Проводит один эксперимент.

    Параметры:
    -----------
    n_points : int
        Количество точек.
    n_dimensions : int, optional
        Размерность пространства (по умолчанию 2D).
    metric : str, optional
        Метрика для вычисления расстояний. По умолчанию 'euclidean'.
    integer_values : bool, optional
        Если True, расстояния округляются до целых чисел. По умолчанию False.
    random_seed : int, optional
        Сид для воспроизводимости результатов. По умолчанию None.

    Возвращает:
    -----------
    results : dict
        Словарь с результатами эксперимента:
        - distance_matrix: исходная матрица расстояний.
        - ultrametric_matrices: список ультраметричных матриц.
        - linkage_matrices: список матриц связей Z.
        - distances: таблица 6x3 с расстояниями (L1, L2, Linf) для каждого метода.
        - random_seed: сид эксперимента.
    """
    # Генерация матрицы расстояний и точек
    distance_matrix, points, random_seed = generate_random_distance_matrix(
        n_points, n_dimensions, metric, integer_values, random_seed
    )

    # Методы кластеризации
    methods = ['minmax', 'median', 'single', 'complete', 'average', 'ward']
    # Словари для хранения результатов
    ultrametric_matrices = {}
    linkage_matrices = {}
    distances = []




    for method in methods:
        if method in ['minmax', 'median']:
            # Используем ваши кастомные функции
            if method == 'minmax':
                Z = custom_linkage(distance_matrix, minimax_distance)
            else:
                Z = custom_linkage(distance_matrix, median_distance)
        else:
            # Используем стандартную функцию linkage
            condensed_matrix = squareform(distance_matrix)
            Z = linkage(condensed_matrix, method)

        # Сохраняем матрицу связей Z
        linkage_matrices[method] = Z

        # Строим ультраметричные матрицы для всех методов
        ultrametric_matrices = build_ultrametric_matrices(linkage_matrices, n_points)

        # Вычисляем расстояния между исходной и ультраметричными матрицами
    for method in methods:
        ultrametric_matrix = ultrametric_matrices[method]

        # Вычисляем расстояния
        dist_L1 = matrix_distance(distance_matrix, ultrametric_matrix, 1)
        dist_L2 = matrix_distance(distance_matrix, ultrametric_matrix, 'fro')
        dist_Linf = matrix_distance(distance_matrix, ultrametric_matrix, norm_type=np.inf)

        # Сохраняем расстояния
        distances.append([method, dist_L1, dist_L2, dist_Linf])

        # Формируем итоговый словарь
    results = {
        'distance_matrix': distance_matrix,
        'ultrametric_matrices': ultrametric_matrices,
        'linkage_matrices': linkage_matrices,
        'distances': np.array(distances),
        'random_seed': random_seed
    }

    return results

def calculate_summary_table(all_results):
    """
    Вычисляет итоговую таблицу 6x9 на основе результатов всех экспериментов.

    Параметры:
    -----------
    all_results : list
        Список словарей с результатами экспериментов.

    Возвращает:
    -----------
    summary_table : dict
        Словарь, где ключи — названия методов, а значения — списки с:
        [L1_mean, L1_var, L1_wins, L2_mean, L2_var, L2_wins, Linf_mean, Linf_var, Linf_wins].
    """
    # Инициализация структур для хранения данных
    methods = ['minmax', 'median', 'single', 'complete', 'average', 'ward']
    metrics = ['L1', 'L2', 'Linf']
    data = {method: {metric: [] for metric in metrics} for method in methods}
    wins = {method: {metric: 0 for metric in metrics} for method in methods}

    # Собираем данные из всех экспериментов
    for result in all_results:
        for row in result['distances']:
            method = row[0]  # Название метода
            L1 = float(row[1])  # Преобразуем в float
            L2 = float(row[2])  # Преобразуем в float
            Linf = float(row[3])  # Преобразуем в float

            data[method]['L1'].append(L1)
            data[method]['L2'].append(L2)
            data[method]['Linf'].append(Linf)

    # Нормализуем данные для каждой метрики отдельно
    for metric in metrics:
        # Собираем все значения для текущей метрики
        all_values = np.concatenate([data[method][metric] for method in methods])
        # Нормализуем все значения
        normalized_values = normalize_data(all_values)
        # Разделяем нормализованные значения обратно по методам
        start_idx = 0
        for method in methods:
            end_idx = start_idx + len(data[method][metric])
            data[method][metric] = normalized_values[start_idx:end_idx]
            start_idx = end_idx

    # Вычисляем количество выигрышей
    for result in all_results:
        for metric_idx, metric in enumerate(metrics):
            try:
                # Находим метод с минимальным расстоянием для текущей метрики
                best_method = min(result['distances'], key=lambda x: float(x[metric_idx + 1]))[0]
                wins[best_method][metric] += 1
            except (ValueError, IndexError) as e:
                print(f"Ошибка в данных: {e}")
                print(f"Результат: {result}")

    # Вычисляем среднее, дисперсию и добавляем количество выигрышей
    summary_table = {}
    for method in methods:
        row = []
        for metric in metrics:
            values = data[method][metric]
            mean = np.mean(values)
            var = np.var(values, ddof=1)  # Используем выборочную дисперсию
            row.extend([mean, var, wins[method][metric]])  # Добавляем среднее, дисперсию и выигрыши

        summary_table[method] = row

    return summary_table

def normalize_data(values):
    """
    Нормализует данные с использованием Min-Max Normalization.

    Параметры:
    -----------
    values : list или np.array
        Список значений для нормализации.

    Возвращает:
    -----------
    normalized_values : np.array
        Нормализованные значения.
    """
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.zeros_like(values)  # Если все значения одинаковы
    return (values - min_val) / (max_val - min_val)

def main():

    n_points = 100
    n_dimensions = 3
    metric = 'euclidean' #'euclidean', 'cityblock', 'cosine', 'chebyshev', 'hamming':
    integer_values = False
    numbersofexp = 1000
    run = True

    """
    Проводит 100 экспериментов и сохраняет результаты.

    Параметры:
    -----------
    n_points : int
        Количество точек.
    n_dimensions : int, optional
        Размерность пространства (по умолчанию 2D).
    metric : str, optional
        Метрика для вычисления расстояний. По умолчанию 'euclidean'.
    integer_values : bool, optional
        Если True, расстояния округляются до целых чисел. По умолчанию False.
    """
    experiment_dir = f"experiment_{metric}_{n_dimensions}D_{n_points}"

    if run == True:
        for i in range(numbersofexp):
            print(f"Эксперимент {i + 1}/{numbersofexp}")
            results = run_experiment(n_points, n_dimensions, metric, integer_values)
            save_experiment(results, experiment_dir, f"experiment_{results['random_seed']}")

    # Загружаем все эксперименты
    all_results = load_experiment(experiment_dir)

    # Вычисляем итоговую таблицу
    summary_table = calculate_summary_table(all_results)

    # Выводим таблицу
    print("Итоговая таблица 6x9:")
    for method, row in summary_table.items():
        print(f"{method}: {row}")

    df = pd.DataFrame.from_dict(summary_table, orient='index',
                                columns=['L1_mean', 'L1_var', 'L1_wins',
                                         'L2_mean', 'L2_var', 'L2_wins',
                                         'Linf_mean', 'Linf_var', 'Linf_wins'])
    print(df)
    df.to_csv("summary_table.csv")
if __name__ == "__main__":
    main()