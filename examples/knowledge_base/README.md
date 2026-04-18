# Knowledge base example — context compression

This example shows how a retrieval tool should behave to be a good citizen of
the context window:

1. **Field projection** — the search response carries `doc_id`, `title`,
   `score`, `tags`, and `fragments`. Not the document body, not internal
   metadata, not API plumbing.
2. **Semantic highlighting** — fragments are ~240-character spans around term
   matches. The LLM sees only the parts that justify the match, in document
   order, instead of the full doc.
3. **Explicit follow-up tool** — `fetch_document` exists for when a fragment
   genuinely isn't enough. The schema tells the agent not to call it
   speculatively.

## Run the demo

```bash
pip install -e ../..
python run.py
```

The demo prints a side-by-side comparison: how many characters the naive
"return every document" approach would emit versus the compressed search
response. Typical reduction for a 6-document corpus answering one query is
80%+ — the savings grow with corpus size.
