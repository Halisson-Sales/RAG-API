from fastapi import FastAPI
from pydantic import BaseModel
import openai
import psycopg2
import os

app = FastAPI()

# =========================
# CONFIG
# =========================

openai.api_key = os.getenv("OPENAI_API_KEY")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# =========================
# REQUEST MODEL
# =========================

class QueryRequest(BaseModel):
    tenant_id: str
    question: str
    match_count: int = 5


# =========================
# RAG ENDPOINT
# =========================

@app.post("/rag")
def rag_query(request: QueryRequest):

    # 1️⃣ Gerar embedding
    embedding_response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )

    query_embedding = embedding_response.data[0].embedding

    # 2️⃣ Conectar no banco
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

    cur = conn.cursor()

    # 3️⃣ Chamar função hybrid_search
    cur.execute(
        """
        select id, content, score, rank
        from hybrid_search(%s, %s, %s, %s)
        """,
        (
            request.tenant_id,
            request.question,
            query_embedding,
            request.match_count
        )
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    # 4️⃣ Se não encontrou nada
    if not results:
        return {
            "answer": "Não encontrei essa informação na base de conhecimento.",
            "sources": []
        }

    # 5️⃣ Montar contexto
    context = "\n\n".join([row[1] for row in results])

    # 6️⃣ Gerar resposta final
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Responda usando apenas o contexto fornecido. Se não houver informação suficiente, diga que não encontrou na base."
            },
            {
                "role": "user",
                "content": f"Contexto:\n{context}\n\nPergunta:\n{request.question}"
            }
        ]
    )

    return {
        "answer": completion.choices[0].message.content,
        "sources": [
            {
                "id": row[0],
                "score": row[2]
            }
            for row in results
        ]
    }
