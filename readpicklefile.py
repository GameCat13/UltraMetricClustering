import pickle

# Открываем файл для чтения в бинарном режиме
with open('experiment_100/experiment_383362625.pkl', 'rb') as f:
    # Загружаем объект из файла
    data = pickle.load(f)

# Теперь data содержит объект, который был сохранен в файле
print(data)