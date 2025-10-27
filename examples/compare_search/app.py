from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import traceback
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ydb.vectorstores import YDB, YDBSettings, YDBSearchStrategy

import os
from dotenv import load_dotenv
load_dotenv()

# Загружаем все переменные окружения один раз
YDB_HOST = os.getenv("YDB_HOST", "localhost")
YDB_PORT = int(os.getenv("YDB_PORT", "2136"))
YDB_DATABASE = os.getenv("YDB_DATABASE", "/local")
YDB_TABLE = os.getenv("YDB_TABLE", "documents")
YDB_INDEX_NAME = os.getenv("YDB_INDEX_NAME", "vector_index")
YDB_FULLTEXT_INDEX_NAME = os.getenv("YDB_FULLTEXT_INDEX_NAME", "fulltext_idx")
YDB_INDEX_LEVELS = int(os.getenv("YDB_INDEX_LEVELS", "3"))
YDB_INDEX_CLUSTERS = int(os.getenv("YDB_INDEX_CLUSTERS", "128"))
YDB_INDEX_TREE_SEARCH_TOP_SIZE = int(os.getenv("YDB_INDEX_TREE_SEARCH_TOP_SIZE", "10"))
YDB_COLUMN_ID = os.getenv("YDB_COLUMN_ID", "id")
YDB_COLUMN_DOCUMENT = os.getenv("YDB_COLUMN_DOCUMENT", "document")
YDB_COLUMN_EMBEDDING = os.getenv("YDB_COLUMN_EMBEDDING", "embedding")
YDB_COLUMN_METADATA = os.getenv("YDB_COLUMN_METADATA", "metadata")

# Инициализация модели эмбеддингов
model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

# Конфигурация YDB из переменных окружения
config = YDBSettings(
    host=YDB_HOST,
    port=YDB_PORT,
    database=YDB_DATABASE,
    column_map={
        "id": YDB_COLUMN_ID,
        "document": YDB_COLUMN_DOCUMENT,
        "embedding": YDB_COLUMN_EMBEDDING,
        "metadata": YDB_COLUMN_METADATA
    },
    strategy=YDBSearchStrategy.COSINE_DISTANCE,
    table=YDB_TABLE,
    index_enabled=True,
    index_name=YDB_INDEX_NAME,
    index_config_levels=YDB_INDEX_LEVELS,
    index_config_clusters=YDB_INDEX_CLUSTERS,
    index_tree_search_top_size=YDB_INDEX_TREE_SEARCH_TOP_SIZE,
)

# Инициализация векторного хранилища
vs = YDB(embedding=model, config=config)

# Функции поиска
def regular_similarity_search(embedding: list[float], k: int = 10):
    return vs.similarity_search_by_vector_with_score(
        embedding=embedding,
        k=k
    )

def hybrid_search(embedding: list[float], tokens: list[str], k: int = 10, max_missing_tokens_count: int = 0):
    return vs._hybrid_search_by_vector_with_score(
        embedding=embedding,
        fulltext_tokens=tokens,
        fulltext_index_table_name=YDB_FULLTEXT_INDEX_NAME,
        k=k,
        max_missing_tokens_count=max_missing_tokens_count
    )

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query: str = data.get('query', '')
        k = int(data.get('k', 10))
        hybrid_enabled = data.get('hybrid_enabled', False)
        max_missing_tokens = int(data.get('max_missing_tokens', 1))

        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Получаем эмбеддинг для запроса
        query_embedding = model.embed_query(query)

        # Токенизируем запрос для hybrid search (простая токенизация по пробелам)
        tokens = query.strip().split()

        # Выполняем поиск с измерением времени
        print(f"Query: {query}")
        print(f"Tokens: {tokens}")
        print(f"K: {k}, Hybrid enabled: {hybrid_enabled}, Max missing tokens: {max_missing_tokens}")

        # Vector search с измерением времени
        vector_start = time.time()
        vector_results = regular_similarity_search(query_embedding, k)
        vector_time = time.time() - vector_start
        print(f"Vector results count: {len(vector_results)}, time: {vector_time:.3f}s")

        # Hybrid search только если включен
        hybrid_results = []
        hybrid_time = 0
        if hybrid_enabled:
            hybrid_start = time.time()
            hybrid_results = hybrid_search(query_embedding, tokens, k, max_missing_tokens)
            hybrid_time = time.time() - hybrid_start
            print(f"Hybrid results count: {len(hybrid_results)}, time: {hybrid_time:.3f}s")
        else:
            print("Hybrid search disabled")

        # Форматируем результаты
        def format_results(results):
            formatted = []
            for doc, score in results:
                formatted.append({
                    'document': doc.page_content,
                    'score': float(score),
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
            return formatted

        response_data = {
            'vector_search': format_results(vector_results),
            'timing': {
                'vector_time': round(vector_time * 1000, 1),  # в миллисекундах
            }
        }

        # Добавляем результаты гибридного поиска только если он был включен
        if hybrid_enabled:
            response_data['hybrid_search'] = format_results(hybrid_results)
            response_data['timing']['hybrid_time'] = round(hybrid_time * 1000, 1)

        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
