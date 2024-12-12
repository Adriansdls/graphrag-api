import os
from local import active_local_search
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import tiktoken

from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from localSearchModified import LocalSearchModified
from lanceModified import LanceDBVectorStore
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Directory for parquet files and embeddings
INPUT_DIR = "./parquets"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2

# Load DataFrames
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
relationships = read_indexer_relationships(relationship_df)
text_units = read_indexer_text_units(text_unit_df)

# Vector store
description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
description_embedding_store.connect(db_uri=LANCEDB_URI)

store_entity_semantic_embeddings(
    entities=entities,
    vectorstore=description_embedding_store
)

# LLM and embeddings
chat_llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

token_encoder = tiktoken.get_encoding("cl100k_base")

text_embedder = OpenAIEmbedding(
    api_key=os.environ["OPENAI_API_KEY"],
    api_base=None,
    api_type=OpenaiApiType.OpenAI,
    model="text-embedding-3-large",
    max_retries=20,
)

def embed_community_reports(
    input_dir: str,
    embedder: OpenAIEmbedding,
    community_report_table: str = COMMUNITY_REPORT_TABLE,
):
    """Embeds the full content of the community reports if not already embedded."""
    input_path = Path(input_dir) / f"{community_report_table}.parquet"
    output_path = Path(input_dir) / f"{community_report_table}_with_embeddings.parquet"

    if output_path.exists():
        return pd.read_parquet(output_path)

    print("Embedding file not found. Computing community report embeddings...")
    report_df = pd.read_parquet(input_path)
    if "full_content" not in report_df.columns:
        raise ValueError(f"'full_content' column not found in {input_path}")

    report_df["full_content_embeddings"] = report_df["full_content"].apply(embedder.embed)
    report_df.to_parquet(output_path)
    print(f"Embeddings saved to {output_path}")
    return report_df

report_df = embed_community_reports(INPUT_DIR, text_embedder)

reports = read_indexer_reports(
    report_df,
    entity_df,
    COMMUNITY_LEVEL,
    content_embedding_col="full_content_embeddings",
)

# Build context and search objects
context_builder = DRIFTSearchContextBuilder(
    chat_llm=chat_llm,
    text_embedder=text_embedder,
    entities=entities,
    relationships=relationships,
    reports=reports,
    entity_text_embeddings=description_embedding_store,
    text_units=text_units,
)
search = DRIFTSearch(
    llm=chat_llm, context_builder=context_builder, token_encoder=token_encoder
)

app = FastAPI()

class QueryModel(BaseModel):
    query: str

@app.post("/drift")
async def retrieve(payload: QueryModel):
    user_query = payload.query
    # Run the async search
    resp = await search.asearch(user_query)
    
    # Extract the answer from the response
    if resp.response["nodes"]:
        answer = resp.response["nodes"][0].get("answer", "No answer found.")
    else:
        answer = "No answer found."
    
    return {"response": answer}

@app.get("/retrieve")
async def retrieve_get(query: str):
    resp = await search.asearch(query)
    answer = resp.response["nodes"][0].get("answer", "No answer found.") if resp.response["nodes"] else "No answer found."
    return {"response": answer}

@app.post("/local")
def retrieve(payload: QueryModel):
    user_query = payload.query
    results = active_local_search.search(user_query)
    return {"response": results.response}

@app.post("/get_data")
def retrieve(payload: QueryModel):
    user_query = payload.query
    results = active_local_search.search(user_query)
    return {"response": results.context_text}

@app.get("/local_get")
def retrieve(query: str):
    results = active_local_search.search(query)
    return {"response": results.context_text}