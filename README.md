# Masala Embed
Serverless Multimodal Food Retrieval for Mood, Weather &amp; Region
---

## Splits (Roles & Leads)
- **Inference Lead (Cost):** Vedant  
- **Infra Lead (Scalability & Latency/Throughput):** Arnav Bharti  
- **Data Lead:** Vinayak  
- **Training Lead (R@1, R@5):** Sahitya Singh  

- **Mentor:**  [Nirant Kasliwal](https://www.linkedin.com/in/nirant/)
- **Advisor:** [Dhruv Anand](https://www.linkedin.com/in/dhruv-anand-ainorthstartech/)
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
## What You Will Learn
- Hands-on **MLOps** (Modal)  
- **Serverless edge deployment** (Cloudflare)  
- Real-world **latency SLOs**

## What You Will Have by Semester End
- Visible, demo-ready **IP to showcase at APOGEE**

## If Everything Goes Right
- Potential to convert into a **top-tier ACL/EMNLP workshop paper**


## Supervision
- **Weekly meetings** with the student  
- Rubric as road-map: **5 hard checkpoints over the semester**  
- Examiner column included for faculty audit  
- **Grade scale: 10 points**
## Checkpoints & Grading Rubric

| Score | Week | High Speed Inference Pipeline | Multi-modal Food Embedding Model | Examiner Verification | Comments / Risk Flags |
|-------|------|-----------------------------|-----------------------------------|------------------------|-----------------------|
| **10 / 10 – Distinction** | Wk 10: Ship & publish | 1. **Query2Dish model** deployed: <300ms p95, <$0.01/1000 requests<br>2. Production SLA: >99% uptime across 50 concurrent queries<br>3. Stripe paywall live; first paid key logged | 1. **Query2Dish benchmark**: R@1 >40%, R@5 >70%<br>2. 10k+ Query2Dish dataset released (mood/weather/region)<br>3. 6-page ACL/EMNLP workshop paper submitted | Public URL; run `wrk` for latency; cost dashboard; leaderboard JSON; repo v1.0; EasyChair receipt; Stripe screenshot | Requires weekly "demo-or-die" discipline; dataset QA is usual downfall |
| **9 / 10 – Very Good** | Wk 9: Production optimize | 1. **Custom Query2Dish model** deployed: <350ms p95, <$0.015/1000 requests<br>2. Auto-deployment pipeline: Modal → GGUF → Workers AI<br>3. Stripe paywall functional with usage analytics | 1. Query2Dish model: R@1 >35%, R@5 >65%<br>2. 8k+ Query2Dish pairs with quality validation<br>3. Beats SigLIP by +5 R@1 on im2recipe benchmark | Same tests with looser thresholds; cost tracking dashboard | Missing piece usually cost optimization on Modal |
| **8 / 10 – Good** | Wk 8: Custom model deploy | 1. **First custom model** deployed: <500ms p95, <$0.02/1000 requests<br>2. End-to-end cost tracking: Modal training + Workers AI serving<br>3. Load testing: 100+ concurrent requests/sec | 1. Query2Dish model trained on 5k+ pairs<br>2. R@1 >30%, R@5 >60% on Query2Dish benchmark<br>3. Beats baseline on Recipe1M+ sampled set | Eval notebook; latency dashboard; cost breakdown; **custom model served** | Data noise or GPU overruns hit here |
| **7 / 10 – Satisfactory** | Wk 6: Integration + training | 1. **Benchmarking pipeline**: im2recipe, recipe2im, Query2Dish<br>2. SigLIP/ColQwen served: <600ms p95, cost tracking setup<br>3. **Integration working**: trained model → deployed via pipeline | 1. Query2Dish dataset: 3k+ curated pairs<br>2. First custom model training on sampled Recipe1M+<br>3. Baseline reproduction: SigLIP on 15k samples | Faculty runs `make eval`; **benchmark results JSON**; API serves trained model | Data cleaning underestimated → common blocker |
| **6 / 10 – Pass** | Wk 4: Benchmarking MVP | 1. **Cost tracking**: Modal training + Workers AI serving costs<br>2. Benchmark pipeline: automated R@1, R@5 evaluation<br>3. SigLIP deployed via Workers AI: <800ms p95 | 1. **Recipe1M+ sampled**: 10-15k representative samples<br>2. Query2Dish schema + initial 500 pairs<br>3. Data preprocessing pipeline for multimodal inputs | Examiner hits `/healthz`; **benchmark runs automatically**; cost dashboard shows $/request | Easy with coding skills; no novelty yet |
| **5 / 10 – Borderline** | Wk 3: Foundation + sampling | 1. **SigLIP/ColQwen endpoint**: basic embeddings via Workers AI<br>2. **Latency baseline**: measure existing model performance<br>3. Modal + Cloudflare deployment setup | 1. **Recipe1M+ sampling strategy**: 5k samples selected<br>2. Query2Dish data collection plan + 100 examples<br>3. Baseline: SigLIP performance on sampled data | PDFs + job screenshot + **working API endpoint** + **sample data verification** | Acceptable only if severe blockers documented |
| **4 / 10 – Bare Minimum** | < Wk 3 | Repo exists, some code committed, no runnable pipeline/data/charter | Recipe1M+ splits which are high diverstiy | Passable |

## Weighting for Final Transcript  
(Agreed with **Prof. Dhruv**)  

| Component | Weight | How Graded |
|-----------|--------|------------|
| **Checkpoint completion** | 70% | Highest fully achieved band = base grade |
| **Technical depth & code quality** | 15% | Code review rubric: modularity, tests, CI, lint, comments |
| **Documentation & replication kit** | 10% | Can an external TA reproduce in < 2h? |
| **Professionalism** | 5% | Weekly demos, logbook, budget tracking (−1 per missed demo, capped −3) |