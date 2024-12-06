import logging
import time
from typing import Any

from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import SearchResult

DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}

log = logging.getLogger(__name__)

class LocalSearchModified(LocalSearch):
    """Search orchestration for local search mode."""

    def get_data(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""

        context_data = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            return SearchResult(
                response="response",
                context_data=context_data,
                context_text="context_text",
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=""
            )

        except Exception:
            log.exception("Exception in get_data")
            return SearchResult(
                response="",
                context_data=context_data,
                context_text="context_text",
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=""
            )