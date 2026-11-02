# AMD AI League — Banksy AI entries

Solutions for the AMD AI League (lablab.ai) mini-challenges. Every entry is a Docker image built on the mandated base image `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` and runs on AMD Instinct (CDNA) or Radeon (RDNA) GPUs.

| Folder | Challenge |
|---|---|
| [`mc3-rag/`](mc3-rag/) | Mini-Challenge 3: retrieval-augmented Q&A over a mixed-filetype corpus, with citations |

## Mini-Challenge 3: RAG with citations

The container is given a folder of mixed documents (PDF, DOCX, XLSX, CSV, logs, Python source, PNG/JPG) and answers ten questions. Each answer is the bare value plus the exact set of files needed to support it, or an empty answer when the corpus doesn't contain it.

**How it works**

1. **Defensive indexing (`--index`).** The corpus is walked in sorted order, catching errors file by file. An unreadable file, an encrypted PDF, an unknown binary or an empty directory is logged and skipped, and the walk continues. Text is extracted per format. Withdrawn or superseded documents are flagged from their filename or title block.
2. **Images become text.** A resident vision-language model (Qwen3-VL-8B-Instruct, bf16, weights baked into the image) transcribes every image once at index time. It writes out relationships such as pin-to-signal mappings, so they can be searched.
3. **Resident model server.** `--index` forks a detached server that loads the model once, builds and saves the index, warms up and listens on a unix socket. Each `--query` call is a thin client, so a question takes about 0.5 s instead of reloading the weights each time.
4. **Retrieval with a second hop.** BM25 over chunks, then a second search on identifiers found in the first hits (e.g. a ticket ID named in a log). That is how two-file, multi-hop answers are found.
5. **Answer, then verify.** The model returns `{answer, sources}` as JSON. Deterministic checks then require:
   - the answer to appear literally in a cited file;
   - withdrawn documents to be dropped when a current document gives the same value.

   A second yes/no check then confirms the evidence is about the same item, attribute and conditions the question asks about. Anything that fails is refused with `{"answer": "", "citations": []}`.

**Measured on 1x MI300X** (sample corpus, no network, `--cap-drop DAC_OVERRIDE`): 10/10 sample questions correct, index pass about 23 s, about 0.5 s per question, VRAM peak about 18.5 GB, image about 45 GiB uncompressed.

**Build and run**

```bash
cd mc3-rag
docker build -t mc3:local .
docker run -d --name mc3 --network none --device=/dev/kfd --device=/dev/dri --group-add video mc3:local
docker cp /path/to/corpus/. mc3:/app/corpus
docker exec mc3 python3 /app/app.py --index /app/corpus
docker exec mc3 python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "What is the maximum junction temperature?"
docker exec mc3 cat /app/output/query_01_output.json
```

Logs (skipped files, image transcriptions, the model's raw replies) are written to `/app/index/server.log` inside the container.

Built by Derek Soltis (Banksy AI).
