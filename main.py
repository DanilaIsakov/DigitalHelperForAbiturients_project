import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")


# Функция для предобработки текста
def preprocess_text_spacy(text):
    if not isinstance(text, str):
        return ''

    # Приведение текста к нижнему регистру
    text = text.lower()

    # Обработка текста с помощью spaCy
    doc = nlp(text)

    # Лемматизация и удаление стоп-слов и знаков препинания
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

    # Объединение лемматизированных токенов обратно в строку
    return ' '.join(lemmatized_tokens)


# Загрузка и подготовка данных
def load_and_prepare_data(file_path):
    # Загрузка данных из Excel
    df = pd.read_excel(file_path, engine='openpyxl')  # Указываем engine для работы с .xlsx файлами

    # Проверка наличия необходимых колонок
    if 'Вопросы' not in df.columns or 'Ответы' not in df.columns:
        raise ValueError("База данных должна содержать колонки 'Вопросы' и 'Ответы'.")

    # Предобработка вопросов
    df['Processed_Questions'] = df['Вопросы'].apply(preprocess_text_spacy)

    return df


# Векторизация вопросов с использованием модели SentenceTransformer
def vectorize_questions(df, model):
    question_embeddings = model.encode(df['Processed_Questions'].tolist(), convert_to_tensor=True)
    return question_embeddings


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

    # Проверка, есть ли схожие вопросы
    if similarities.max() < 0.6:  # Порог схожести теперь 0.6
        return ["Извините, ответа на данный вопрос еще нет."], [similarities.max()]

    # Получение индекса наиболее похожего вопроса
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Получение ответов
    answers = df.iloc[top_indices]['Ответы'].values
    scores = similarities[top_indices]

    return answers, scores



# Главная функция
def main():
    # Определение пути к директории, в которой находится скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Построение относительного пути к файлу базы данных
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


def load_model_and_data():
    # Построение пути к файлу базы данных
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'DB_DigitalHelper.xlsx')
    try:
        df = load_and_prepare_data(file_path)

        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        question_embeddings = vectorize_questions(df, model)

        return df, model, question_embeddings
    except Exception as e:
        print(f"Произошла ошибка при подготовке данных: {e}")
        return None, None, None


def ask_question(user_question, df, model, question_embeddings):
    answers, scores = find_answer(user_question, df, model, question_embeddings)
    for i, (answer, score) in enumerate(zip(answers, scores), 1):
        return f"\nОтвет {i} (Схожесть: {score:.4f}):\n{answer}"


if __name__ == "__main__":
    main()
