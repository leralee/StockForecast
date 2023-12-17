# Python
FROM python:latest

# Установка рабочей директории
WORKDIR /micro_3_modeling

COPY requirements.txt .

# Копирование файла requirements.txt и установка зависимостей
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


 #pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения в контейнер
COPY micro_3_modeling/ .

# Порт, который будет использоваться микросервисом
EXPOSE 5003

# Запуск приложения при старте контейнера
CMD ["python3", "app.py"]