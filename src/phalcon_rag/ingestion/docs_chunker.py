from ..models import Chunk, Document
from .chunking.utils import _word_count, MAX_CHUNK_WORDS
from .chunking.headings import _split_by_headings
from .chunking.api import _contains_api_methods, _split_by_api_methods, _split_method_annotations, _contains_method_annotations, _contains_api_items, _split_by_api_items

from .chunking.tables import _is_markdown_table, _split_large_markdown_table, _contains_events_table, _split_events_table
from .chunking.code import _split_by_method_signatures, _contains_method_signatures, _contains_multiple_method_signatures, _split_method_signature_list

from .chunking.examples import _is_query_builder_examples, _split_query_builder_examples
from .chunking.fallback import _split_oversized_chunk
from .chunking.filters import _filter_chunks


class DocsChunker:

    def chunk(self, document: Document) -> list[Chunk]:
        heading_chunks = _split_by_headings(document)

        chunks: list[Chunk] = []

        for chunk in heading_chunks:
            if _contains_api_methods(chunk.content):
                chunks.extend(_split_by_api_methods(chunk))

            elif _contains_events_table(chunk):
                chunks.extend(_split_events_table(chunk))

            elif (_word_count(chunk.content) > MAX_CHUNK_WORDS and _contains_method_signatures(chunk.content)):
                chunks.extend(_split_by_method_signatures(chunk))

            else:
                chunks.append(chunk)

        chunks = _filter_chunks(chunks)

        oversized_split_chunks: list[Chunk] = []

        for chunk in chunks:
            if _word_count(chunk.content) > MAX_CHUNK_WORDS:
                oversized_split_chunks.extend(_split_oversized_chunk(chunk))
            else:
                oversized_split_chunks.append(chunk)

        final_chunks: list[Chunk] = []

        for chunk in oversized_split_chunks:
            if _word_count(chunk.content) <= MAX_CHUNK_WORDS:
                final_chunks.append(chunk)

            elif _is_query_builder_examples(chunk):
                final_chunks.extend(_split_query_builder_examples(chunk))

            elif _contains_api_items(chunk.content):
                final_chunks.extend(_split_by_api_items(chunk))

            elif _is_markdown_table(chunk.content):
                final_chunks.extend(_split_large_markdown_table(chunk))

            elif _contains_multiple_method_signatures(chunk.content):
                final_chunks.extend(_split_method_signature_list(chunk))
           
            elif _contains_method_annotations(chunk.content):
                final_chunks.extend(_split_method_annotations(chunk))

            else:
                final_chunks.append(chunk)

        return final_chunks

    