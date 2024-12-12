import os
import pandas as pd
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from localSearchModified import LocalSearchModified
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from lanceModified import LanceDBVectorStore

# Load environment variables
load_dotenv()

# Set configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o-mini"


# Adjust INPUT_DIR if needed. Ensure these files are present in the deployed code.
INPUT_DIR = "./parquets"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 3

# Load DataFrames
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
relationships = read_indexer_relationships(relationship_df)
claims = read_indexer_covariates(covariate_df)
covariates = {"claims": claims}
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
text_units = read_indexer_text_units(text_unit_df)

# Set up vectorstore
description_embedding_store = LanceDBVectorStore(
    collection_name="entity_description_embeddings",
)
description_embedding_store.connect(db_uri=LANCEDB_URI)

store_entity_semantic_embeddings(
    entities=entities,
    vectorstore=description_embedding_store
)

# Set up LLM and embedding
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_CHAT_MODEL,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

token_encoder = tiktoken.get_encoding("cl100k_base")

text_embedder = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model=OPENAI_EMBED_MODEL,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=covariates,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,
    text_embedder=text_embedder,
    token_encoder=token_encoder
)

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 0,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 5,
    "top_k_relationships": 5,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": True,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,
    "max_tokens": 12000
}

llm_params = {
    "max_tokens": 2000,
    "temperature": 0.0,
}

active_local_search = LocalSearchModified(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="educational content that is engaging and interesting for the reader",
    system_prompt="Use only the following content to answer the question"
)

# app = FastAPI()

# class QueryModel(BaseModel):
#     query: str

# @app.post("/drift")
# def retrieve(payload: QueryModel):
#     user_query = payload.query
#     results = active_local_search.get_data(user_query)
#     return {"response": results.response}

# @app.post("/local")
# def retrieve(payload: QueryModel):
#     user_query = payload.query
#     results = active_local_search.get_data(user_query)
#     return {"response": results.response}
