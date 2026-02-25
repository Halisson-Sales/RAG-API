import os
import json
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ==============================
# MODELS
# ==============================

class IngestRequest(BaseModel):
    tenant_id: str
    documents: List[str]
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = 5


# ==============================
# DATABASE CONNECTION
# ==============================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

# ==============================
# INGEST ENDPOINT
# ==============================

@app.post("/ingest")
def ingest_documents(request: IngestRequest):

    try:
        conn = get_connection()
        cur = conn.cursor()

        for doc in request.documents:

            # Gerar embedding
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )

            embedding = embedding_response.data[0].embedding

            # Inserir no banco
            cur.execute(
                """
                INSERT INTO documents (tenant_id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    request.tenant_id,
                    doc,
                    embedding,
                    json.dumps(request.metadata) if request.metadata else None
                )
            )

        conn.commit()
        cur.close()
        conn.close()

        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# QUERY ENDPOINT
# ==============================

@app.post("/query")
def query_documents(request: QueryRequest):

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Gerar embedding da pergunta
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )

        query_embedding = embedding_response.data[0].embedding

        # Busca vetorial filtrada por tenant
        cur.execute(
            """
            SELECT content,
                   embedding <=> %s::vector AS distance
            FROM documents
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_embedding,
                request.tenant_id,
                query_embedding,
                request.top_k
            )
        )

        results = cur.fetchall()

        cur.close()
        conn.close()

        documents = [
            {"content": row[0], "distance": row[1]}
            for row in results
        ]

        return {"results": documents}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def health():
    return {"status": "API online"}


