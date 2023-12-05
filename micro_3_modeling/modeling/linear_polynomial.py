__all__ = ['LinearPolynomial']

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from datetime import timedelta

from math import floor,ceil,sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import timedelta
from numpy.random import seed



class LinearPolynomial:
    def __init__(self, data, dates, column):
        self.data = data
        self.dates = dates
        self.column = column
        self.shape = data.shape[0]

        print(self.shape)
        print(self.data)
        print(self.dates)
        print(self.column)

        # Разделение данных на обучающую и тестовую выборки
        train_size = int(0.8 * len(data))
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]
        self.train_dates = dates[:train_size]
        self.test_dates = dates[train_size:]

        # Извлечение признаков (дни с начала временного ряда) и значений
        self.X_train = (self.train_dates - self.train_dates.min()).dt.days.values.reshape(-1, 1)
        self.y_train = self.train_data[column].values
        self.X_test = (self.test_dates - self.test_dates.min()).dt.days.values.reshape(-1, 1)
        self.y_test = self.test_data[column].values

    # def calc_model(self):
    #     min_degree = 1  # Минимальная степень полинома
    #     max_degree = 5  # Максимальная степень полинома
    #     best_rmse = float('inf')  # Начальное значение RMSE
    #     best_degree = None  # Начальное значение степени полинома
    #
    #     for degree in range(min_degree, max_degree + 1):
    #         poly = PolynomialFeatures(degree=degree)
    #         X_train_poly = poly.fit_transform(self.X_train)
    #         X_test_poly = poly.transform(self.X_test)
    #
    #         model = LinearRegression()
    #         model.fit(X_train_poly, self.y_train)
    #
    #         y_pred = model.predict(X_test_poly)
    #         rmse = sqrt(mean_squared_error(self.y_test, y_pred))
    #
    #         if rmse < best_rmse:
    #             best_rmse = rmse
    #             best_degree = degree
    #
    #     return [best_degree, best_rmse]

    def predict(self):
        shape = self.shape
        # df_new=df[['Close']]
        df_new = self.data
        df_new.head()
        dataset = df_new.values
        train = df_new[:ceil(shape * 0.75)]
        valid = df_new[ceil(shape * 0.75):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        x_train, y_train = [], []
        for i in range(40, len(train)):
            x_train.append(scaled_data[i - 40:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        # Подготовка входных данных для валидационной выборки
        inputs = df_new[len(df_new) - len(valid) - 40:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        # Подготовка X_test для валидационной выборки
        X_test = []
        for i in range(40, inputs.shape[0]):
            X_test.append(inputs[i - 40:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Прогнозирование цены закрытия
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        # Вычисление RMSE
        rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))

        print('RMSE value on validation set:', rms)
        print('-----------------------------------------------------------')
        print('-----------------------------------------------------------')

        # Добавление прогнозов в DataFrame для валидационной выборки
        valid['Predictions'] = closing_price

        # Подготовка входных данных для прогнозирования на 30 дней вперед
        last_40_days_scaled = scaled_data[-40:]

        # Предсказание на 30 дней вперед
        future_predictions = np.empty((0, 1))  # Создаем пустой массив NumPy

        for i in range(365):
            X_last_40_days = last_40_days_scaled.reshape(1, -1, 1)
            future_price = model.predict(X_last_40_days)

            # Добавляем предсказание в массив
            future_predictions = np.append(future_predictions, future_price[0, 0].reshape(1, 1), axis=0)

            # Обновление входных данных
            last_40_days_scaled = np.append(last_40_days_scaled[1:], future_price).reshape(-1, 1)

        # Обратное масштабирование предсказанных цен
        future_predictions = scaler.inverse_transform(future_predictions)

        return future_predictions


