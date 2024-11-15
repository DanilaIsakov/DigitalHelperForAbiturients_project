import pandas as pd
from transformers import pipeline
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
URL = 'C:/Users/danil/OneDrive - УрФУ/DF_DigitalHelperForAbiturients.xlsx'
df = pd.read_excel(URL)

# Убедитесь, что все значения в 'Вопросы' - строки, и обработайте NaN
df['Вопросы'] = df['Вопросы'].fillna('').astype(str)

# Инициализация модели
device = 0 if torch.cuda.is_available() else -1
embedding_model = pipeline("feature-extraction", model="distilbert-base-cased", device=device)

# Функция для получения эмбеддинга текста
def get_text_embedding(text):
    embedding = embedding_model(text)
    return np.mean(embedding[0], axis=0).flatten()

# Применяем функцию для создания эмбеддингов для каждого вопроса
df['Question Embeddings'] = df['Вопросы'].apply(get_text_embedding)

# Преобразование вопроса пользователя в эмбеддинг
user_question = "сколько бюджетных мест?"
user_vector = get_text_embedding(user_question)

# Сравнение с использованием косинусного сходства
similarities = cosine_similarity([user_vector], df['Question Embeddings'].tolist())
best_match_index = np.argmax(similarities)
best_answer = df['Ответы'].iloc[best_match_index]

print("Наиболее релевантный ответ:", best_answer)
print("Схожесть:", similarities[0][best_match_index])


df.to_excel('DF_DigitalHelperForAbiturients.xlsx', index=False)
