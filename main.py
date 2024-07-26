import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib

matplotlib.use('TkAgg')  # Используйте бэкенд TkAgg
import matplotlib.pyplot as plt

# Инициализация модели логистической регрессии
model = LogisticRegression()
X_train = np.array(
    [[22, 25000], [25, 50000], [47, 80000], [52, 110000], [46, 25000], [56, 100000], [55, 110000], [60, 105000]])
y_train = np.array([0, 0, 1, 1, 0, 1, 1, 1])
model.fit(X_train, y_train)


def update_and_predict(age, income, bought):
    global X_train, y_train, model
    # Добавляем новые данные в тренировочный набор
    new_data = np.array([[age, income]])
    new_label = np.array([bought])
    X_train = np.vstack([X_train, new_data])
    y_train = np.append(y_train, new_label)

    # Обновляем модель
    model.fit(X_train, y_train)

    # Предсказываем на основе новой модели
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0, 1]

    return prediction, probability


# Функция для визуализации
def plot_model():
    plt.figure(figsize=(10, 6))

    # Отображение точек данных
    for i, label in enumerate(y_train):
        if label == 0:
            plt.scatter(X_train[i, 0], X_train[i, 1], color='blue', edgecolors='k', s=100)
        else:
            plt.scatter(X_train[i, 0], X_train[i, 1], color='red', edgecolors='k', s=100)

    # Создание сетки для визуализации разделяющей гиперплоскости
    xx, yy = np.meshgrid(np.linspace(20, 70, 200), np.linspace(20000, 120000, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Контурный график для вероятности покупки
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap='bwr', alpha=0.1)

    plt.xlabel('Возраст')
    plt.ylabel('Доход')
    plt.title('Логистическая Регрессия: Вероятность покупки товара')
    plt.colorbar(label='Вероятность покупки')
    plt.show()




# Визуализация модели с обновленными данными
plot_model()

while True:
    # Пример использования
    age = int(input())
    income = int(input())
    bought = bool(input())
    prediction, probability = update_and_predict(age, income, bought)
    print(f"Предсказание: {'Купит' if prediction == 1 else 'Не купит'}, Вероятность: {probability * 100:.2f}%")
    plot_model()