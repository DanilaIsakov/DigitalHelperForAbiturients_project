import asyncio
from aiogram import Bot, Dispatcher, types, Router, F
import logging
from concurrent.futures import ProcessPoolExecutor
from main import load_model_and_data
from main import ask_question

API_TOKEN = '7519686216:AAEFQaJBNEVK6nmYFXDWXm8fat0pDkTY_SI'
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()
logging.basicConfig(level=logging.INFO)


df, model, question_embeddings = load_model_and_data()

if df is None or model is None or question_embeddings is None:
    print("Ошибка при загрузке модели или данных. Завершение работы.")
    exit(1)

print('Model and data loaded')


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

executor_pool = ProcessPoolExecutor(max_workers=2)


@router.message(lambda message: "неверный ответ" in message.text.lower())
async def wrong_answer(message: types.Message):
    user_message = message.text
    user_id = message.from_user.id
    with open('user_messages.txt', 'a', encoding='utf-8') as file:
        file.write(f"User ID: {user_id}, Message: {user_message}\n\n")

    await message.answer('Прошу прощения за неполноту базы данных, вопрос будет добавлен в скором времени.')


@router.message()
async def handle_message(message: types.Message):
    user_message = message.text
    user_id = message.from_user.id
    answer = ask_question(user_message, df, model, question_embeddings)

    with open('user_messages.txt', 'a', encoding='utf-8') as file:
        file.write(f"User ID: {user_id}, Message: {user_message}\n")
        file.write(f"Answer: {answer}\n\n")

    answer += '\n\nЕсли ответ неудовлетворительный, напишите, пожалуйста, "Неверный ответ".'

    await message.answer(answer)


if __name__ == "__main__":
    asyncio.run(main())
