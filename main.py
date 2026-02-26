import os
import json
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", 5432)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Hybrid RAG API")

class IngestRequest(BaseModel):
    tenant_id: str
    documents: List[str]
    metadata: Optional[dict] = None

class QueryRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = 5

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

@app.post("/ingest")
def ingest_documents(request: IngestRequest):

    try:
        conn = get_connection()
        cur = conn.cursor()

        for doc in request.documents:

            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )

            embedding = embedding_response.data[0].embedding

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

        # Peso dinâmico
        if len(request.query.split()) <= 3:
            vector_weight = 0.5
            text_weight = 0.5
        else:
            vector_weight = 0.7
            text_weight = 0.3

        cur.execute(
            """
            SELECT
                content,
                embedding <=> %s::vector AS vector_distance,
                ts_rank(fts, plainto_tsquery('portuguese', %s)) AS text_rank,
                (
                    (embedding <=> %s::vector) * %s
                    -
                    ts_rank(fts, plainto_tsquery('portuguese', %s)) * %s
                ) AS hybrid_score
            FROM documents
            WHERE tenant_id = %s
            ORDER BY hybrid_score ASC
            LIMIT %s;
            """,
            (
                query_embedding,
                request.query,
                query_embedding,
                vector_weight,
                request.query,
                text_weight,
                request.tenant_id,
                request.top_k
            )
        )

        results = cur.fetchall()

        cur.close()
        conn.close()

        documents = [
            {
                "content": row[0],
                "vector_distance": row[1],
                "text_rank": row[2],
                "hybrid_score": row[3]
            }
            for row in results
        ]

        return {"results": documents}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "API online"}
