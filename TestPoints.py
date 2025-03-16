import seaborn as sns
from matplotlib import pyplot as plt

from input import*
from visualization import*


dimen = 1

distance_matrix, points = generate_random_distance_matrix(n_points=100, n_dimensions=dimen)

print("Матрица расстояний:")
print(distance_matrix)

print("Точки:")
print(points)

if False:
    # Построение тепловой карты с правильными подписями
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix, annot=True, cmap='viridis', cbar=True)
    plt.yticks(rotation=0)
    plt.title("Матрица расстояний")
    plt.xlabel("Номера точек")
    plt.ylabel("Номера точек")
    plt.show()


plot_points(points)
