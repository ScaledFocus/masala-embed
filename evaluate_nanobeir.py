#!/usr/bin/env python3
"""
NanoBEIR evaluator (robust llama.cpp client).

- Optional baseline via Sentence-Transformers
- llama.cpp --embedding HTTP server (/embedding)

Outputs per dataset:
- R@1, R@5, R@10, NDCG@10, MedR
- Latency p95 (ms) on ~50 single-query embeds
- Cost per 1k queries (from config)
"""

import os, json, math, time, statistics, argparse
from typing import List, Dict, Any, Iterable, Union

# ----------------- Tunables -----------------
MAX_CHARS = 2000          # hard limit for doc text sent to /embedding
RETRY_BACKOFF = [0.0, 0.2, 0.6]   # seconds; 3 tries total
RETRY_SHRINK  = [1.00, 0.60, 0.35]  # shrink content length on retries

# ----------------- Small utilities -----------------
def p95(latencies_ms: List[float]) -> float:
    if not latencies_ms:
        return 0.0
    return float(statistics.quantiles(latencies_ms, n=100)[94])

def time_calls(fn, payloads: List[Any], warmup: int = 3) -> float:
    if not payloads:
        return 0.0
    for _ in range(max(0, warmup)):
        _ = fn([payloads[0]])
    lat = []
    for p in payloads:
        t0 = time.time()
        _ = fn([p])
        t1 = time.time()
        lat.append((t1 - t0) * 1000.0)
    return p95(lat)

def median_rank(results, qrels) -> float:
    ranks = []
    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue
        relevant = {did for did, rel in qrels[qid].items() if rel > 0}
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (did, _) in enumerate(ranked, start=1):
            if did in relevant:
                ranks.append(idx)
                break
    if not ranks:
        return 0.0
    ranks.sort()
    m = len(ranks)
    return float(ranks[m//2] if m % 2 else (ranks[m//2 - 1] + ranks[m//2]) / 2)

# ----------------- Minimal BEIR imports -----------------
def _imports():
    global util, GenericDataLoader, EvaluateRetrieval, DRES
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# ----------------- Dataset URLs -----------------
_BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa":    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
}

def load_dataset(name: str, base_dir: str = "./datasets"):
    _imports()
    url_or_id = _BEIR_URLS.get(name, name)
    data_path = util.download_and_unzip(url_or_id, base_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

# ----------------- Model wrappers -----------------
def _doc_text(doc: Dict[str, str]) -> str:
    title = (doc.get("title") or "").strip()
    text  = (doc.get("text")  or "").strip()
    merged = (title + " " + text).strip()
    if len(merged) > MAX_CHARS:
        merged = merged[:MAX_CHARS]
    return merged

class STModel:
    def __init__(self, hf_id: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(hf_id)

    def encode_queries(self, queries, batch_size=32, **kwargs):
        return self.model.encode(
            queries, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False
        )

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        docs_iter = corpus.values() if isinstance(corpus, dict) else corpus
        texts = [
            ((doc.get("title") or "") + " " + (doc.get("text") or "")).strip()[:2000]
            for doc in docs_iter
        ]
        return self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True
        )


class ColQwenPooledEncoder:
    """
    Calls a ColQwen-style endpoint that returns token-level embeddings per text,
    then mean-pools them to a single vector so we can use cosine retrieval.

    Expected server response for POST { "text": "..." } (or {"texts":[...]}) is one of:
      - {"tokens": [[f1,...],[f2,...], ...]}     # a list of token vectors
      - {"data": [{"tokens": [[...], ...]}], ...}
      - {"embeddings": [[...], ...]}  # if it already returns tokens directly
    """
    def __init__(self, endpoint: str):
        import requests
        self.r = requests
        self.endpoint = endpoint.rstrip("/")

    def _fetch_tokens(self, text: str, timeout=60):
        resp = self.r.post(self.endpoint, json={"text": text}, timeout=timeout)
        resp.raise_for_status()
        js = resp.json()
        # Try common shapes
        if isinstance(js, dict):
            if "tokens" in js:
                return js["tokens"]
            if "data" in js and js["data"]:
                item = js["data"][0]
                if isinstance(item, dict) and "tokens" in item:
                    return item["tokens"]
            if "embeddings" in js:
                return js["embeddings"]
        raise ValueError(f"Unexpected ColQwen tokens response: {str(js)[:200]}")

    @staticmethod
    def _mean_pool(tokens_2d):
        # tokens_2d: list[list[float]] -> single list[float]
        if not tokens_2d:
            return []
        dim = len(tokens_2d[0])
        sums = [0.0]*dim
        for t in tokens_2d:
            # skip malformed tokens
            if not isinstance(t, (list, tuple)) or len(t) != dim:
                continue
            for i, v in enumerate(t):
                sums[i] += float(v)
        n = max(1, len(tokens_2d))
        vec = [s/n for s in sums]
        # L2-normalize
        import math
        nrm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/nrm for x in vec]

    def _embed_one(self, text: str):
        toks = self._fetch_tokens(text)
        return self._mean_pool(toks)

    def encode_queries(self, queries, batch_size=1, **kwargs):
        return [self._embed_one(q) for q in queries]

    def encode_corpus(self, corpus, batch_size=1, **kwargs):
        def doc_text(d):
            title = (d.get("title") or "").strip()
            body  = (d.get("text")  or "").strip()
            x = (title + " " + body).strip()
            return x[:2000] if len(x) > 2000 else x
        docs_iter = corpus.values() if isinstance(corpus, dict) else corpus
        return [self._embed_one(doc_text(d)) for d in docs_iter]


class LlamaServerEncoder:
    """Client for llama.cpp --embedding server (/embedding).  Ensures fixed-dim vectors."""
    def __init__(self, endpoint: str):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.2, status_forcelist=(502, 503, 504))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.endpoint = endpoint.rstrip("/")
        self._dim = None  # lock the embedding size after first good response

    def _parse_embedding_json(self, js):
        """
        Make a best effort to dig out the first numeric vector from many shapes:
        - {"embedding":[...]} or {"embedding":[[...]]}
        - {"data":[{"embedding":[...]}]}
        - [{"index":0,"embedding":[[...]]}, ...]
        - {"embeddings":[...]} or [[...]] or [...]
        """
        def first_vector(obj):
            # unwrap dicts by common keys
            if isinstance(obj, dict):
                for k in ("embedding", "vector", "values", "data"):  # try typical fields
                    if k in obj:
                        return first_vector(obj[k])
                # if dict holds a list under any key, dive into the first value
                if obj:
                    return first_vector(next(iter(obj.values())))
                return []

            # unwrap lists: if nested lists/dicts, keep diving until we hit numbers
            if isinstance(obj, (list, tuple)):
                if not obj:
                    return []
                # if it's a list of dicts (e.g., [{"index":0,"embedding":[[...]]}])
                if isinstance(obj[0], dict):
                    return first_vector(obj[0])
                # if it's a list of lists, take the first inner list
                if isinstance(obj[0], (list, tuple)):
                    return first_vector(obj[0])
                # otherwise, assume it's already a flat numeric list
                return obj

            # scalar fallback
            return [obj]

        return first_vector(js)

    def _to_1d_numeric(self, obj):
        """Coerce whatever we got into a flat list[float]."""
        # flatten recursively in case anything nested remains
        def flatten(xs):
            for x in xs:
                if isinstance(x, (list, tuple)):
                    yield from flatten(x)
                elif isinstance(x, dict):
                    # common sub-keys first
                    for k in ("vector", "embedding", "values", "data"):
                        if k in x:
                            yield from flatten(x[k])
                            break
                    else:
                        for v in x.values():
                            yield from flatten(v)
                else:
                    yield x

        vec = list(flatten(self._parse_embedding_json(obj)))
        try:
            vec = [float(x) for x in vec]
        except Exception:
            raise ValueError(f"Embedding not numeric; sample: {str(vec)[:120]}")
        return vec

    def _normalize_vec(self, vec):
        vec = self._to_1d_numeric(vec)

        # fix target dim from first success
        if self._dim is None:
            self._dim = len(vec)
            if self._dim < 64:
                raise ValueError(f"Embedding dimension too small: {self._dim}")
            print(f"[llama] locked embedding dim = {self._dim}", flush=True)

        # pad / truncate to fixed dim
        if len(vec) < self._dim:
            vec = vec + [0.0] * (self._dim - len(vec))
        elif len(vec) > self._dim:
            vec = vec[:self._dim]
        return vec

    def _embed_one(self, text: str, timeout=60):
        orig = text
        for backoff, shrink in zip(RETRY_BACKOFF, RETRY_SHRINK):
            if backoff: time.sleep(backoff)
            t = orig[: int(MAX_CHARS * shrink)]
            r = self.session.post(self.endpoint, json={"content": t}, timeout=timeout)
            if r.status_code >= 500:
                continue
            r.raise_for_status()
            vec = self._parse_embedding_json(r.json())
            return self._normalize_vec(vec)
        # final tiny attempt
        t = orig[: min(512, len(orig))]
        r = self.session.post(self.endpoint, json={"content": t}, timeout=timeout)
        r.raise_for_status()
        vec = self._parse_embedding_json(r.json())
        return self._normalize_vec(vec)

    def encode_queries(self, queries, batch_size=1, **kwargs):
        return [self._embed_one(q) for q in queries]

    def encode_corpus(self, corpus, batch_size=1, **kwargs):
        docs_iter = corpus.values() if isinstance(corpus, dict) else corpus
        texts = []
        for i, doc in enumerate(docs_iter, 1):
            title = (doc.get("title") or "").strip()
            text  = (doc.get("text")  or "").strip()
            merged = (title + " " + text).strip()
            if len(merged) > MAX_CHARS:
                merged = merged[:MAX_CHARS]
            texts.append(merged)
            if i % 500 == 0:
                print(f"[llama] prepared {i} docs...", flush=True)

        embs = []
        for i, t in enumerate(texts, 1):
            embs.append(self._embed_one(t))
            if i % 500 == 0 or i == len(texts):
                print(f"[llama] encoded {i}/{len(texts)} docs", flush=True)
        return embs





# ----------------- Evaluation -----------------
def evaluate_model(beir_model, corpus, queries, qrels, cost_per_1k: float):
    _imports()
    retriever_model = DRES(beir_model, batch_size=32)
    retriever = EvaluateRetrieval(retriever_model, score_function="cos_sim")

    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    R1  = float(recall.get(1,  recall.get("1",  0.0)))
    R5  = float(recall.get(5,  recall.get("5",  0.0)))
    R10 = float(recall.get(10, recall.get("10", 0.0)))
    N10 = float(ndcg.get(10,   ndcg.get("10",   0.0)))
    medr = float(median_rank(results, qrels))

    # latency p95 on first 50 queries
    qids = list(queries.keys())
    probe = [queries[q] for q in qids[:min(50, len(qids))]]
    lat = time_calls(lambda batch: beir_model.encode_queries(batch, batch_size=1), probe, warmup=3)

    return {
        "R@1": R1, "R@5": R5, "R@10": R10,
        "NDCG@10": N10, "MedR": medr,
        "Latency_ms_p95": float(lat),
        "Cost_per_1k_queries": float(cost_per_1k),
    }

# ----------------- Main -----------------
def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    leaderboard_path = cfg.get("leaderboard_json", f"{out_dir}/nanobeir_leaderboard.json")

    datasets = cfg.get("datasets", ["scifact"])
    cost = cfg.get("cost_per_1k", {})
    models_cfg = cfg.get("models", {})

    all_out: Dict[str, Dict[str, Any]] = {}

    for ds in datasets:
        corpus, queries, qrels = load_dataset(ds)
        all_out.setdefault(ds, {})

        if "baseline" in models_cfg:
            m_id = models_cfg["baseline"]["hf_id"]
            print(f"[baseline] {m_id}")
            out = evaluate_model(STModel(m_id), corpus, queries, qrels, cost.get("baseline", 0.0))
            all_out[ds]["baseline"] = out

        if "qwen_edge_llamaserver" in models_cfg:
            ep = models_cfg["qwen_edge_llamaserver"]["endpoint"]
            print(f"[qwen_edge_llamaserver] {ep}")
            out = evaluate_model(LlamaServerEncoder(ep), corpus, queries, qrels,
                                 cost.get("qwen_edge_llamaserver", 0.0))
            all_out[ds]["qwen_edge_llamaserver"] = out

    with open(leaderboard_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"[saved] {leaderboard_path}")
    print(json.dumps(all_out, indent=2))

if __name__ == "__main__":
    main()
