__all__ = ['LinearPolynomial']

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import floor,ceil,sqrt
from datetime import timedelta


class LinearPolynomial:
    def __init__(self, data, dates, column):
        self.data = data
        self.dates = dates
        self.column = column
        self.shape = data.shape[0]

        # Разделение данных на обучающую и тестовую выборки
        train_size = int(0.8 * len(data))
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]
        self.train_dates = dates[:train_size]
        self.test_dates = dates[train_size:]

        # Извлечение признаков (дни с начала временного ряда) и значений
        self.X_train = (self.train_dates - self.train_dates.min()).dt.days.values.reshape(-1, 1)
        self.y_train = self.train_data[column].values
        self.X_test = (self.test_dates - self.train_dates.min()).dt.days.values.reshape(-1, 1)
        self.y_test = self.test_data[column].values


    def calc_model(self):
        min_degree = 1  # Минимальная степень полинома
        max_degree = 4  # Максимальная степень полинома
        best_rmse = float('inf')  # Начальное значение RMSE
        best_degree = None  # Начальное значение степени полинома
        best_model = None

        for degree in range(min_degree, max_degree + 1):
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(self.X_train)
            X_test_poly = poly.transform(self.X_test)

            model = LinearRegression()
            model.fit(X_train_poly, self.y_train)

            y_pred = model.predict(X_test_poly)

            rmse = sqrt(mean_squared_error(self.y_test, y_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_degree = degree

        return [best_degree, best_rmse]

    def predict(self, best_degree):
        # # Обучение модели с оптимальной степенью полинома
        poly = PolynomialFeatures(degree=best_degree)
        X_train_poly = poly.fit_transform(self.X_train)

        model = LinearRegression()
        model.fit(X_train_poly, self.y_train)

        # Генерация прогноза на месяц вперед
        last_date = self.dates.max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 365)]
        X_future = (np.array([(date - self.train_dates.min()).days for date in future_dates])).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        y_future = model.predict(X_future_poly)

        return y_future




