import logging
import time
from typing import Any

from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import SearchResult

log = logging.getLogger(__name__)

class LocalSearchModified(LocalSearch):
    """Modified LocalSearch that does not generate LLM responses."""

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Build local search context that fits a single context window, but do not generate an answer from the LLM."""
        start_time = time.time()
        search_prompt = ""
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}

        # Build the local context without invoking the LLM for an answer
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )

        # Record metrics related to context building
        llm_calls["build_context"] = context_result.llm_calls
        prompt_tokens["build_context"] = context_result.prompt_tokens
        output_tokens["build_context"] = context_result.output_tokens

        try:
            # Return the gathered context without LLM response generation
            return SearchResult(
                response="",  # no LLM output
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=sum(llm_calls.values()),
                prompt_tokens=sum(prompt_tokens.values()),
                output_tokens=sum(output_tokens.values()),
                llm_calls_categories=llm_calls,
                prompt_tokens_categories=prompt_tokens,
                output_tokens_categories=output_tokens,
            )
        except Exception:
            log.exception("Exception in LocalSearchModified search")
            # In case of an exception, return partial context and empty response
            return SearchResult(
                response="",
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=0,
            )
