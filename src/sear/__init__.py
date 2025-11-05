"""
SEAR: Summarization-Enhanced Augmented Retrieval
Multi-corpus RAG with FAISS retrieval, boolean queries, SQL interface, and LLM summarization
"""

from .core import (
    # Main functions
    index_file,
    extract_relevant_content,
    execute_query,
    execute_sql_query,
    parse_sql_query,
    delete_corpus,
    list_corpuses,
    get_corpus_info,
    search,

    # Boolean operations
    union_results,
    intersect_results,
    difference_results,

    # Document ordering
    sort_by_document_order,
    merge_adjacent_chunks,

    # GPU support
    is_gpu_available,
    get_gpu_info,
)

from .cli import main

__version__ = "2.3.0"

__all__ = [
    # Main API
    "index_file",
    "extract_relevant_content",
    "execute_query",
    "execute_sql_query",
    "parse_sql_query",
    "delete_corpus",
    "list_corpuses",
    "get_corpus_info",
    "search",

    # Boolean operations
    "union_results",
    "intersect_results",
    "difference_results",

    # Document ordering
    "sort_by_document_order",
    "merge_adjacent_chunks",

    # GPU support
    "is_gpu_available",
    "get_gpu_info",

    # CLI
    "main",
]
