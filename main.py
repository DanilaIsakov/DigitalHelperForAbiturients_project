import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import time

# Функция для предобработки текста
def preprocess_text_spacy(text):
    if not isinstance(text, str):
        return ''
    
    # Приведение текста к нижнему регистру
    text = text.lower()

    # Удаление ненужных символов (простой вариант)
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    return text


# Загрузка и подготовка данных
def load_and_prepare_data(file_path):
    # Загрузка данных из Excel
    df = pd.read_excel(file_path, engine='openpyxl')

    # Проверка наличия необходимых колонок
    if 'Вопросы' not in df.columns or 'Ответы' not in df.columns:
        raise ValueError("База данных должна содержать колонки 'Вопросы' и 'Ответы'.")

    # Предобработка вопросов
    df['Processed_Questions'] = df['Вопросы'].apply(preprocess_text_spacy)

    return df


# Векторизация вопросов с использованием модели SentenceTransformer
def vectorize_questions(df, model):
    return model.encode(df['Processed_Questions'].tolist(), convert_to_tensor=True)


# Поиск ответа на вопрос пользователя
def find_answer(user_question, df, model, question_embeddings, top_k=1):
    # Предобработка пользовательского вопроса
    processed_question = preprocess_text_spacy(user_question)

    if not processed_question:
        return ["Пожалуйста, введите корректный вопрос."], [0.0]

    # Векторизация
    user_embedding = model.encode([processed_question], convert_to_tensor=True)

    # Вычисление косинусной похожести
    similarities = cosine_similarity(user_embedding, question_embeddings)[0]

    # Проверка на схожесть
    if similarities.max() < 0.6:  # Порог схожести 0.6
        return ["Извините, я не могу найти подходящего ответа на ваш вопрос."], [similarities.max()]

    # Получение индекса наиболее похожего вопроса
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Получение ответов
    answers = df.iloc[top_indices]['Ответы'].values
    scores = similarities[top_indices]

    return answers, scores


# Главная функция
def main():
    # Построение пути к файлу базы данных
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'DB_DigitalHelper.xlsx')

    try:
        # Загрузка и подготовка данных
        print("Загрузка и подготовка данных...")
        df = load_and_prepare_data(file_path)

        # Загрузка модели SentenceTransformer для векторизации
        start = time.time()
        print("Загрузка модели...")
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # Замените на доступную модель
        print(f"Модель загружена за {time.time() - start:.2f} секунд.")

        # Векторизация вопросов
        print("Векторизация вопросов...")
        question_embeddings = vectorize_questions(df, model)

        print("Система готова к использованию.")
    except Exception as e:
        print(f"Произошла ошибка при подготовке данных: {e}")
        return

    while True:
        # Ввод пользовательского вопроса
        user_question = input("\nВведите ваш вопрос (или 'выход' для завершения): ")
        if user_question.lower() in ['выход', 'exit', 'quit']:
            print("Завершение работы.")
            break

        # Поиск ответа
        answers, scores = find_answer(user_question, df, model, question_embeddings)

        # Вывод ответа
        for i, (answer, score) in enumerate(zip(answers, scores), 1):
            print(f"\nОтвет {i} (Схожесть: {score:.4f}):\n{answer}")


if __name__ == "__main__":
    main()
