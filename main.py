from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
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

# Conexão Postgres
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT
)

class QueryRequest(BaseModel):
    query: str
    match_count: int = 10

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

    # 2️⃣ Chamar função Postgres
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM hybrid_search(
            %s,
            %s,
            %s
        )
        """,
        (query_text, query_embedding, match_count)
    )

    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    response = [
        dict(zip(columns, row))
        for row in results
    ]

    return {"results": response}