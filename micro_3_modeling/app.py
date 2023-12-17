from flask import Flask, request, jsonify
import requests
import pandas as pd
import modeling as mdl
import os

app = Flask(__name__)

@app.route('/getprediction', methods=['GET'])
def getprediction():
    dataset = request.args.get('dataset', 'close_price_stationary')
    column = request.args.get('column', '')
    if not column:
        jsonify(None)

    datasets_base_url = os.getenv('DATASETS_URL', 'http://localhost:5002')
    # Загрузка дат и данных компании
    response = requests.get(f'{datasets_base_url}/getcolumn?dataset={dataset}&column={column}')
    # Получаем данные в формате JSON
    json_data = response.json()
    df = pd.DataFrame({
        'dates': json_data[0],
        column: json_data[1]
    })
    df['dates'] = pd.to_datetime(df['dates'])

    data = df[[column]]
    dates = df['dates']

    model = mdl.LinearPolynomial(data, dates, column)
    best_degree, best_rmse = model.calc_model()
    y_future = model.predict(best_degree)
    last_val = y_future[0]
    return [best_degree, best_rmse, y_future.tolist(), last_val]


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = os.getenv('PORT', 5003)
    port = int(port)
    app.run(host=host, port=port, debug=True)