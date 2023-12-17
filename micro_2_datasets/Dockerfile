# Python
FROM python:latest

# Установка рабочей директории
WORKDIR /micro_2_datasets

# Копирование файла requirements.txt и установка зависимостей
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения в контейнер
COPY /micro_2_datasets .

# Порт, который будет использоваться микросервисом
EXPOSE 5002

# Запуск приложения при старте контейнера
CMD ["python3", "app.py"]