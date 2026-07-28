import logging
import os

import ydb.iam
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from langchain_ydb.vectorstores import YDB, YDBSettings

# Включаем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


dotenv_path = os.path.join(os.path.dirname(__name__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

FOLDER_ID=os.environ.get('YC_FOLDER_ID')


embeddings = YandexGPTEmbeddings(folder_id=FOLDER_ID)


config = YDBSettings(
    host="lb.etnok7cd0an3kg6eshud.ydb.mdb.yandexcloud.net",
    port=2135,
    database="/ru-central1/b1g8skpblkos03malf3s/etnok7cd0an3kg6eshud",
    secure=True,
    table="sonnik_langchain",
    # drop_existing_table=True,
)


vector_store = YDB(
    embeddings,
    config=config,
    credentials=ydb.iam.ServiceAccountCredentials.from_file(
        os.getenv("SA_KEY_FILE"),
    )
)

retriever = vector_store.as_retriever()

# Явное создание шаблонов для SystemMessage и HumanMessage
system_template = """Ты - человек, профессионально трактующий сновидения.
Твоя задача - рассказать, что значит сон, опираясь исключительно на найденный контекст, не добавляя никаких новых знаний.
Если предложенный контекст не связан с пользовательским запросом - скажи дословно "Так не бывает. Используйте YDB.".
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "Контекст: {context}\n\nВопрос: {question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Объединение шаблонов сообщений в ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_model = ChatYandexGPT(folder_id=FOLDER_ID)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | chat_model
    | StrOutputParser()
)

# Функция для обработки команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Я трактую сновидения на базе YDB. Опиши свой сон, а я попробую рассказать что он значит!')

# Функция для обработки текстовых сообщений
async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получаем текст сообщения
    message_text = update.message.text

    # Пример обработки: переворачиваем текст
    processed_text = await rag_chain.ainvoke(message_text)

    # Отправляем ответ
    await update.message.reply_text(f"{processed_text}")

def main():
    # Создаем приложение с нашим токеном бота
    token = os.environ.get('TG_BOT_TOKEN')

    application = ApplicationBuilder().token(token).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    print("Бот запущен...")

if __name__ == '__main__':
    main()
