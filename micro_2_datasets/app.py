from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/getdataset', methods=['GET'])
def getdataset():
    # Отправляет весь датасет. Неэффективно, так как большой трафик
    dataset = request.args.get('dataset', 'close_price_stationary')
    df = pd.read_csv(f'datasets/{dataset}.csv', delimiter='\t')
    return jsonify(df.to_dict(orient='records'))

@app.route('/getcolumns', methods=['GET'])
def getcolumns():
    # Отправляет названия колонок датасета
    dataset = request.args.get('dataset', 'close_price_stationary')
    df = pd.read_csv(f'datasets/{dataset}.csv', delimiter='\t')
    columns = df.columns[1:].to_list()
    return jsonify(columns)

@app.route('/getcolumn', methods=['GET'])
def getcolumn():
    # Отправляет названия колонок датасета
    dataset = request.args.get('dataset', 'close_price_stationary')
    column = request.args.get('column', '')
    if not column:
        jsonify(None)
    df = pd.read_csv(f'datasets/{dataset}.csv', delimiter='\t')
    df.columns.values[0] = "dates"
    data = df[column].to_list()
    dates = df['dates'].to_list()
    return jsonify([dates, data])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=True)