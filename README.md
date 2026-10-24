# RAG-Eval

This project is a retrieval-augmented generation (RAG) evaluation pipeline for
the transcript files in [`data/`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/data).
It can run for free with Ollama locally. Groq is also supported as an
OpenAI-compatible hosted model when a Groq API key is available.

## How the flow works

1. **Load transcripts** — [`src/retriver.py`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/src/retriver.py)
   reads the `.vtt` files, removes WebVTT timestamps, and adds the session
   number as document metadata.
2. **Split and embed** — each transcript is split into overlapping chunks.
   The default embedding provider is the local Hugging Face model
   `sentence-transformers/all-MiniLM-L6-v2`, so no embedding API key is needed.
   Ollama embeddings can be selected instead.
3. **Persist in Chroma** — the chunks and vectors are stored in
   [`chroma_store/`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/chroma_store).
   The store is reused on later runs. If it is empty, it is built automatically.
4. **Retrieve** — Chroma returns the most similar chunks for the question.
5. **Rerank** — [`src/reranker.py`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/src/reranker.py)
   over-retrieves candidates and ranks them with the
   `cross-encoder/ms-marco-MiniLM-L-6-v2` model.
6. **Generate** — [`src/generator.py`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/src/generator.py)
   sends the question and top context chunks to either Ollama or Groq. The
   prompt requires the answer to stay grounded in the retrieved context.
7. **Return the result** — [`src/rag_pipeline.py`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/src/rag_pipeline.py)
   returns the query, retrieved context, and generated answer.

## Setup

Use the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the environment has not been created yet, install the dependencies using
the existing lock file:

```powershell
uv sync
```

The first run may download the local Hugging Face embedding and reranker
models. Internet access is only needed for those model downloads.

## Run locally with Ollama

Install and start Ollama, then download the generator and embedding models:

```powershell
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
```

The supplied `.env` is configured for this free local flow. Run the complete
pipeline from the repository root:

```powershell
python -m src.rag_pipeline
```

You can also test retrieval without generation:

```powershell
python src/retriver.py
```

## Run with Groq

Create a free Groq API key and set it in `.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
```

Then run:

```powershell
python -m src.rag_pipeline
```

Groq is used only for generation. Embeddings and reranking remain local.

## Run the evaluations

The evaluation scripts use the same retrieval and generation components:

```powershell
python -m evals.eval_retriver
python -m evals.eval_generator
python -m evals.eval_rag_pipeline
```

Evaluation uses the local Ollama model configured by `EVAL_JUDGE_MODEL`, so it
does not consume Groq tokens. The default is the installed `qwen2.5:0.5b`.
Keep Ollama running before starting an evaluation. Evaluation is slower than
the normal pipeline because every test case is judged by the model.

To use Groq for evaluation instead, set:

```dotenv
EVAL_JUDGE_PROVIDER=groq
EVAL_JUDGE_MODEL=openai/gpt-oss-20b
```

Groq free-tier rate limits can interrupt larger evaluation runs. Switch back
to `EVAL_JUDGE_PROVIDER=ollama` for a fully local run.

## Configuration

The main settings are in [`.env`](C:/Users/VISHAL%20BHAGAT/Desktop/RAG-Eval/.env):

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `ollama`, `groq`, or `auto` |
| `OLLAMA_MODEL` | Ollama generation model |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model |
| `OLLAMA_BASE_URL` | Ollama server URL |
| `EMBEDDING_PROVIDER` | `huggingface` or `ollama` |
| `HUGGINGFACE_EMBEDDING_MODEL` | Local Hugging Face embedding model |
| `GROQ_API_KEY` | Required only when `LLM_PROVIDER=groq` |
| `GROQ_MODEL` | Groq generation model |

`LLM_PROVIDER=ollama` always uses Ollama locally. `LLM_PROVIDER=groq` uses
Groq and requires `GROQ_API_KEY`. With `LLM_PROVIDER=auto`, Groq is selected
when `GROQ_API_KEY` is set; otherwise Ollama is selected. If Groq reports
`model_not_found`, update `GROQ_MODEL` to a model currently listed by your
Groq account, or switch back to `LLM_PROVIDER=ollama`.

If you change the embedding provider or model after the Chroma store has
already been created, remove the contents of `chroma_store/` and run the
pipeline again so the vectors are rebuilt with the matching model.