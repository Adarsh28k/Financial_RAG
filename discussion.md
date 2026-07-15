# Building this RAG system — the problems I ran into and how I fixed them

This is a rough log of everything that went wrong while I put this pipeline together, in the order I hit it. Some of these took way longer to track down than they should have, so I'm writing it down mostly for future-me.


## 1. "What is apple?" returned "Apple Inc." and nothing else

Once the segfault was gone, I actually started looking at result quality, and this one was funny in a bad way. I asked "what is apple?" and the single top (and only) result was a chunk whose entire text was `"Apple Inc."` — no other content, just the company name sitting under a "SIGNATURES" heading in the 10-K.

Went and looked at the actual source markdown, and the signature block of the filing looked like this:

```
Pursuant to the requirements of Section 13 or 15(d)...

Date: October 31, 2025

Apple Inc.

By:
```

Every one of those blank-line-separated lines was becoming its *own* chunk, because my chunker had no concept of a minimum chunk size — it just split on blank lines and called it a day. A two-word chunk like `"Apple Inc."` embeds as basically a pure keyword vector with zero surrounding context, so it scores unnaturally high against a short query that happens to contain that exact keyword. It was winning purely because it had nothing else competing for the embedding's attention.

Fixed it by rewriting the chunker to merge consecutive short paragraphs together instead of emitting them as standalone chunks — so that whole signature block now becomes one coherent chunk instead of four useless fragments. While I was in there I also noticed the old chunking code did three separate passes (build table/image chunks, then paragraph chunks, then try to guess how to interleave them back into document order) and the interleaving step was fragile — it assumed roughly one output chunk per input paragraph, which breaks the moment you start merging or splitting unevenly. Rewrote it as a single forward pass over the blocks instead, so ordering is just guaranteed by construction.

Also found, completely by accident while debugging this: the `region` metadata field on every chunk (things like `"Italy"`, `"Sweden"`) isn't real. It's just cycling through a hardcoded list of 20 countries based on chunk index. Not derived from the document at all. Left it alone for now since nothing depends on it, but it's misleading if you don't know that going in.

## 2. Fixed the chunking, and then every query stopped returning anything

This one scared me for a minute. I regenerated the chunks and rebuilt the embeddings/index after the chunking fix, and suddenly *every single query* — not just the apple one — came back with zero results. Even boring, should-definitely-work queries like "what is profit in 2022 and 2023?" that had worked fine minutes earlier.

Turned out to be a completely different problem that the chunking fix accidentally exposed. `retreiver.py` embeds queries using a local `all-MiniLM-L12-v2` model. But `embeddings.py` — which I'd just re-run to pick up the new chunks — was embedding all the chunks with `all-MiniLM-L6-v2` downloaded fresh from Hugging Face. Two different models. Both happen to output 384-dimensional vectors, so nothing errored — FAISS just silently compared vectors from two unrelated embedding spaces and got near-meaningless similarity scores back, which then failed the cosine threshold and got filtered out.

The old index that "worked" must have been built with the matching L12 model before I touched anything. The moment I re-ran `embeddings.py` with its own hardcoded model name, it silently swapped the whole vector space out from under the retriever.

Fix: pointed `embeddings.py` at the exact same local L12 model path the retriever uses, and rebuilt. Lesson learned — the embedding model used at index-build time and query time absolutely have to match, and nothing was checking that they did.

## 3. Reranking was just cosine + a threshold, and it had blind spots

Asked "what is artificial intelligence" and got back nothing, even though the document mentions AI six separate times. Went and checked — every mention was just a passing phrase inside a much longer paragraph about something else entirely (product liability risk, patent risk, that kind of thing). The best matching chunk only scored 0.40 cosine similarity, and I had a hard `COSINE_THRESHOLD = 0.50` gate sitting in front of reranking, so it got thrown out before anything else even got a chance to look at it.

Makes sense in hindsight — a bi-encoder pools the whole chunk into one vector, so if "artificial intelligence" is one clause out of a 900-character paragraph about legal risk, that one clause barely moves the needle on the chunk's overall embedding.

This is exactly the kind of thing a cross-encoder is good at, so I added one (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second reranking stage. Cosine still does the first-pass narrowing (now loosened to 0.30 instead of 0.50), but then the cross-encoder actually scores the real (query, passage) pair jointly instead of comparing two independently-pooled vectors. Dropped the old cosine + lexical-overlap blend entirely since the cross-encoder does a genuinely better job of the same task.

## 4. Reorganizing for more than one company

Everything so far had been Apple-only, flat files sitting directly in `documents/`. Before adding a second company I split things into proper subfolders — `pdf/`, `markdown/`, `images/`, `chunks/`, `embeddings/`, `index/` — and updated all four pipeline scripts to read/write the right subfolder while still taking the same single folder argument on the CLI. Had to fix the image references inside the markdown too, since images and markdown files were no longer sitting next to each other.

## 5. Adding Google exposed a three-way split Python environment

Went to convert `GOOGLE_FINANCE.pdf` and discovered `docling` wasn't installed anywhere I'd been running things from. Went looking and found it in a completely separate Anaconda Python install. Meanwhile `faiss`/`torch`/`sentence-transformers` lived under yet another plain system Python. Three different environments, none of which had everything. Got the pipeline working per-stage by just pointing each script at whichever environment actually had what it needed, but it was clearly not sustainable — flagged it for later.

## 6. A compound question would only ever answer half of it

Asked "what is googles revenue in 2024 and what is apple revenue in 2024" and got back three chunks — all Google, no Apple at all, despite Apple being explicitly named in the question. Global top-3 reranking has no idea the question is actually two questions about two different companies; it just returns whoever scored marginally higher, and Google edged out Apple across the board that time.

Fixed this at the `rag_chat.py` orchestration layer: detect every document mentioned in the query (matching against the known `document` metadata values), and if more than one is mentioned, retrieve each one separately and merge the results, instead of ever letting them compete in the same ranked list. If only one company (or none) is mentioned it behaves exactly like before.


## 7. Wiring up an actual chat step

Last piece was hooking a local LLM up to the retrieval pipeline. Went to test the existing `call_ollama` function and it turned out to have never worked — it was calling `ollama predict`, which isn't a real Ollama subcommand. Would have failed the very first time anyone tried to use it. Switched it to hit Ollama's local REST API instead (`/api/chat`), which is both more robust than shelling out to a CLI and actually correct.

Pulled `llama3.1:8b` and wired up a simple chat loop in the notebook — ask a question, get an answer grounded in whatever got retrieved, keep going until you type "end." Deliberately kept it stateless across turns for now — each question gets retrieved and answered fresh, no memory of what was asked before. Simpler, and it matches how the rest of the pipeline already works.

