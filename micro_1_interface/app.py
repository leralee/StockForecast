from flask import Flask, render_template, request
import numpy as np
import requests
import os

app = Flask(__name__)


def get_columns(dataset: str) -> list:
    datasets_base_url = os.getenv('DATASETS_URL', 'http://localhost:5002')

    # ЧТЕНИЕ ДАТАСЕТА ИЗ ВТОРОГО МИКРОСЕРВИСА
    try:
        # Загрузка данных при старте микросервиса из второго микросервиса
        response = requests.get(f'{datasets_base_url}/getcolumns?dataset={dataset}')
        # Получаем данные в формате JSON
        json_data = response.json()
        return json_data
    except Exception:
        return []


def get_prediction(dataset: str, column: str) -> list:
    modeling_base_url = os.getenv('MODELING_BASE_URL', 'http://localhost:5003')
    # ОБРАЩЕНИЕК К ТРЕТЬЕМУ МИКРОСЕРВИСУ
    try:
        # Загрузка параметров лучшей модели и предсказания для выбранной компании
        response = requests.get(f'{modeling_base_url}/getprediction?dataset={dataset}&column={column}')
        # Получаем данные в формате JSON
        json_data = response.json()
        return json_data
    except Exception:
        return []


@app.route('/', methods=['GET', 'POST'])
def index():
    columns = get_columns('close_price_stationary')
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns[]')
        degrees = {}
        forecasts = {}
        start_periods = {}
        end_periods = {}
        results = {}

        for column in selected_columns:
            best_degree, best_rmse, y_future, last_val = get_prediction('close_price_stationary', column)
            degrees[column] = best_degree
            forecasts[column] = [0 if val < 0 else val for val in list(y_future)]
            start = round(last_val, 2)
            end = round(y_future[-1], 2)
            end = 0 if end < 0 else end
            start_periods[column] = start
            end_periods[column] = end

            if start > end:
                prcnt = round(((start - end) / start) * 100, 2)
                results[column] = f'Цена акций снизится с {start} до {end} (на {prcnt}%)'
            elif start < end:
                prcnt = round(((end - start) / start) * 100, 2)
                results[column] = f'Цена акций вырастет {start} до {end} (на {prcnt}%)'
            else:
                results[column] = f'Цена акций не изменилась'

            # Усредненный прогноз по всем выбранным столбцам
        averaged_forecast = np.mean([forecasts[column] for column in selected_columns], axis=0)

        start_ave = round(np.mean(list(start_periods.values())), 2)
        end_ave = round(averaged_forecast[-1], 2)
        if start_ave > end_ave:
            prcnt = round(((start_ave - end_ave) / start_ave) * 100, 2)
            result_ave = f'Цена портфеля акций снизится с {start_ave} до {end_ave} (на {prcnt}%)'
        elif start_ave < end_ave:
            prcnt = round(((end_ave - start_ave) / start_ave) * 100, 2)
            result_ave = f'Цена портфеля акций вырастет с {start_ave} до {end_ave} (на {prcnt}%)'
        else:
            result_ave = f'Цена портфеля акций не изменилась'

        return render_template('index.html', columns=columns, degrees=degrees,
                               selected_columns=selected_columns,
                               forecasts=forecasts, averaged_forecast=averaged_forecast,
                               results=results, result_ave=result_ave)

    return render_template('index.html', columns=columns)


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = os.getenv('PORT', 5001)
    port = int(port)
    app.run(host=host, port=port, debug=True)
