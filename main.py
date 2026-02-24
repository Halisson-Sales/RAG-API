from fastapi import FastAPI
from pydantic import BaseModel
from psycopg2 import pool
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

app = FastAPI()

# 🔐 Variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# 🔥 Connection Pool (PRODUÇÃO)
connection_pool = pool.SimpleConnectionPool(
    1, 20,  # mínimo e máximo de conexões
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT
)

class QueryRequest(BaseModel):
    query: str
    match_count: int = 5

@app.post("/search")
def hybrid_search(request: QueryRequest):
    query_text = request.query
    match_count = request.match_count

    # 1️⃣ Gerar embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )

    query_embedding = embedding_response.data[0].embedding

    # 2️⃣ Buscar no banco usando pool
    conn = connection_pool.getconn()
    cursor = conn.cursor()

    cursor.execute(
    """
    SELECT * FROM hybrid_search(
        %s,
        %s,
        %s,
        %s,
        0.7,
        1.3,
        40
    )
    """,
    (tenant, query_text, query_embedding, match_count)
)

    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    cursor.close()
    connection_pool.putconn(conn)

    response = [
        dict(zip(columns, row))
        for row in results
    ]

    # 🔥 Retorna contexto pronto para o Agent
    context_text = "\n\n".join(
        [f"Documento {i+1}:\n{r['content']}" for i, r in enumerate(response)]
    )

    return {
        "context": context_text
    }

