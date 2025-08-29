# Masala Embed
Serverless Multimodal Food Retrieval for Mood, Weather &amp; Region

Here’s a clean, structured **Markdown version** of your Google Doc:

```markdown
# Masala Edge
**Serverless Multimodal Food Retrieval for Mood, Weather & Region**

---

## Use Case
Imagine asking:  
> “Need a cheesy, rainy-day dosa”  

and getting an instant, mouth-watering shortlist—whether you’re in **Pilani or Palo Alto**.  

Today’s search and embedding models cannot handle such phrases or sentences.  

**MasalaEdge** will be the **first edge-deployed, triplet-tuned model** that fuses text, images, mood, and weather to power **sub-300 ms search**.  

- **Training:** GPU-heavy Modal jobs  
- **Serving:** Quantized to GGUF → Cloudflare Workers AI  
- **Target:** Beat **SigLIP and CLIP** by +8 R@1 on Recipe1M+  
- **New Data:** 20k **Indian-flavour triplets** (ignored by academia & industry so far)

---

## What You Will Learn
- Hands-on **MLOps** (Modal)  
- **Serverless edge deployment** (Cloudflare)  
- Real-world **latency SLOs**

## What You Will Have by Semester End
- Visible, demo-ready **IP to showcase at APOGEE**

## If Everything Goes Right
- Potential to convert into a **top-tier ACL/EMNLP workshop paper**

---

## Supervision
- **Weekly meetings** with the student  
- Rubric as road-map: **5 hard checkpoints over the semester**  
- Examiner column included for faculty audit  
- **Grade scale: 10 points**

---

## Checkpoints & Grading Rubric

| Score | Week | Deliverables | Examiner Verification | Comments / Risk Flags |
|-------|------|--------------|------------------------|-----------------------|
| **10 / 10 – Distinction** | Wk 10: Ship & publish | ① Edge demo ≤ 300 ms p95 across 50 queries (Workers AI → ColQwen fallback)<br>② Model beats SigLIP-B/16 by ≥ +8 R@1 on Recipe1M+ & Indian split<br>③ ≥ 20k triplets dataset released (CC-BY, license, card)<br>④ 6-page ACL/EMNLP workshop paper submitted<br>⑤ Stripe paywall live; first paid key logged | Public URL; run `wrk` for latency; leaderboard JSON; repo v1.0; EasyChair receipt; Stripe screenshot | Requires weekly “demo-or-die” discipline; dataset QA is usual downfall |
| **9 / 10 – Very Good** | Wk 9: Fine-tuned & paywalled | Latency SLO met (≤ 350 ms) or fallback; recall uplift ≥ +5 R@1 on both datasets; Stripe paywall functional | Same tests with looser thresholds | Missing piece usually latency tuning on Modal |
| **8 / 10 – Good** | Wk 8: Fine-tuned, no paywall | Model beats baseline on Recipe1M+ by ≥ +5 R@1; optional Indian split; Edge demo live (unauthenticated allowed) | Eval notebook; latency > 400 ms acceptable; Stripe not needed | Data noise or GPU overruns hit here |
| **7 / 10 – Satisfactory** | Wk 6: Data + baseline | ✓ 5k curated triplets<br>✓ Baseline (SigLIP) reproduced on Recipe1M+<br>✓ Vector inference service ready<br>✓ README reproducibility steps | Faculty runs `make eval` (±1 pt MedR); spot-checks 50 triplets | Data cleaning underestimated → common blocker |
| **6 / 10 – Pass** | Wk 4: Infra MVP | ① Modal script trains model → quantizes → uploads GGUF to R2 ([example](https://huggingface.co/qwen/qwen3-embedding-0.6b))<br>② Workers AI endpoint returns embeddings<br>③ GitHub Actions CI green | Examiner hits `/healthz`; checks CI log | Easy with coding skills; no novelty yet |
| **5 / 10 – Borderline** | Wk 3: Proposal & sandbox | Charter, timeline, risk register, cost sheet, hello-world Modal job | PDFs + job screenshot | Acceptable only if severe blockers documented |
| **4 / 10 – Bare Minimum** | < Wk 3 | Repo exists, some code committed, no runnable pipeline/data/charter | Faculty checks repo | Passable, but hurts recommendation letters |

---

## Weighting for Final Transcript  
(Agreed with **Prof. Dhruv**)  

| Component | Weight | How Graded |
|-----------|--------|------------|
| **Checkpoint completion** | 70% | Highest fully achieved band = base grade |
| **Technical depth & code quality** | 15% | Code review rubric: modularity, tests, CI, lint, comments |
| **Documentation & replication kit** | 10% | Can an external TA reproduce in < 2h? |
| **Professionalism** | 5% | Weekly demos, logbook, budget tracking (−1 per missed demo, capped −3) |

---

## Splits (Roles & Leads)
- **Inference Lead (Cost):** Vedant  
- **Infra Lead (Scalability & Latency/Throughput):** Arnav Bharti  
- **Data Lead:** Vinayak  
- **Training Lead (R@1, R@5):** Sahitya Singh  
```