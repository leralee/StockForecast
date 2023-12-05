# Python
FROM python:latest

# Установка рабочей директории
WORKDIR /micro_1_interface

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения в контейнер
COPY micro_1_interface/ .

# Порт, который будет использоваться микросервисом
EXPOSE 5001

# Запуск приложения при старте контейнера
CMD ["python3", "app.py"]