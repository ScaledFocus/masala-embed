# Masala Embed: Branch Cleanup & Merge Plan

## User Preferences (Confirmed)
- **Deployment**: Modal only (skip Cloudflare Workers for now)
- **Database**: CSV files only (no PostgreSQL)
- **Credentials Available**: Modal API token, OpenAI/GPT API key
- **Cleanup Strategy**: Archive only (no deletions)

---

## Current State Assessment

### Branch Inventory (14 remote branches)

| Branch | Status | Content Summary | Action |
|--------|--------|-----------------|--------|
| `main` | Current | README + esci-dataset (PR #9 merged) | Base |
| `feature/dataset/initial-generation` | **Merged** | Core query generation pipeline | Done |
| `dataset` | **Merged** | Dataset matching scripts | Done |
| `infrastructure` | **Ready to merge** | Modal inference (skip Cloudflare/Terraform) | Merge (partial) |
| `infra` | Superseded | Basic Modal/FAISS (subset of infrastructure) | Archive |
| `benchmarking` | **Ready to merge** | Performance testing + timing instrumentation | Merge |
| `feature/dataset/annotation-and-autolabel` | **Ready to merge** | Full annotation pipeline + 5000+ Query2Dish pairs | Merge |
| `feature/dataset/annotation-backup` | Backup | Snapshot of annotation work | Archive |
| `finetune` | **Ready to merge** | SigLIP training pipeline + server | Merge |
| `NanoBIER` / `nanobier` | **Ready to merge** | NanoBEIR benchmarking framework | Merge |
| `feat/dataset` | Obsolete | Early experimental work | Archive |
| `saxena-vinayak36-patch-1` | Obsolete | Early README edits | Archive |

---

## Current Grade Assessment: **6/10 (Pass) → Potentially 7/10**

### What's Achieved (Across Branches):
| Rubric Item | Status | Evidence |
|-------------|--------|----------|
| Query2Dish schema + pairs | 5000+ pairs | annotation-and-autolabel branch |
| Benchmark pipeline (R@1, R@5) | Ready | NanoBEIR branch |
| Data preprocessing pipeline | Complete | esci-dataset/src/ |
| SigLIP/ColQwen deployment | Ready but unverified | infrastructure branch |
| Cost tracking | Partial | infrastructure branch |
| Custom model training | Ready but unverified | finetune branch |
| Latency <600ms p95 | Needs testing | benchmarking branch |

### Missing for 7/10:
- [ ] Deployed and verified SigLIP/ColQwen endpoint
- [ ] Benchmark results JSON (R@1, R@5 numbers)
- [ ] Cost tracking dashboard
- [ ] Integration test: trained model → deployed via pipeline

### Missing for 8/10:
- [ ] Custom trained model deployed
- [ ] Load testing: 100+ concurrent requests/sec
- [ ] R@1 >30%, R@5 >60% on Query2Dish benchmark

---

## Recommended Merge Strategy (Simplified for Modal + CSV)

### Phase 1: Infrastructure Foundation
```bash
# 1. Merge infrastructure branch (Modal components only)
git checkout main
git merge origin/infrastructure --no-ff -m "Merge infrastructure: Modal inference pipeline"

# Skip Cloudflare/Terraform directories if present (can remove later)
```

**Files to keep from infrastructure:**
- `infrastructure/scripts/inference.py` - Modal inference server
- `infrastructure/scripts/quantize.py` - Model quantization
- `.env.*` files for model configs

**Files to skip/remove:**
- `infrastructure/worker/` - Cloudflare Workers (not needed)
- `infrastructure/terraform/` - R2 bucket setup (not needed)

### Phase 2: Benchmarking
```bash
# 2. Merge benchmarking branch
git merge origin/benchmarking --no-ff -m "Merge benchmarking: Performance testing"
```

**Adds:**
- `benchmarking/benchmark_configurable.py` - Load testing
- `benchmarking/generate_queries.py` - Test data
- Timing headers in server code

### Phase 3: Dataset & Annotation
```bash
# 3. Merge annotation branch (CSV mode)
git merge origin/feature/dataset/annotation-and-autolabel --no-ff -m "Merge annotation: 5000+ Query2Dish pairs"
```

**Adds:**
- 5000+ Query2Dish pairs (CSV format)
- Annotation tools (Flask apps)
- Data generation scripts (DSPy + GPT)

**Note:** Will use CSV mode, no PostgreSQL needed.

### Phase 4: Training & Evaluation
```bash
# 4. Merge finetune branch
git merge origin/finetune --no-ff -m "Merge finetune: SigLIP training pipeline"

# 5. Merge NanoBIER branch
git merge origin/NanoBIER --no-ff -m "Merge NanoBIER: Benchmarking framework"
```

### Phase 5: Archive Obsolete Branches
```bash
# Archive (rename with prefix) instead of delete
git push origin origin/infra:refs/heads/archive/infra
git push origin origin/feat/dataset:refs/heads/archive/feat-dataset
git push origin origin/feature/dataset/annotation-backup:refs/heads/archive/annotation-backup
git push origin origin/saxena-vinayak36-patch-1:refs/heads/archive/saxena-vinayak36-patch-1
```

---

## Required Setup (Simplified)

### Environment Variables:
```bash
# .env file
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
OPENAI_API_KEY=your_openai_key  # For data generation
```

### Modal Deployment:
```bash
# Deploy inference endpoint
modal deploy infrastructure/scripts/inference.py
```

---

## Post-Merge Validation Checklist

### 1. Infrastructure Validation:
```bash
# Deploy to Modal
cd infrastructure && modal deploy scripts/inference.py

# Test endpoint health
curl -X GET https://your-modal-endpoint/health

# Run latency benchmark
uv run python benchmarking/benchmark_configurable.py
```

### 2. Dataset Validation:
```bash
# Verify Query2Dish pairs count
cd esci-dataset && make summary
# Expected: 5000+ pairs
```

### 3. Model Evaluation:
```bash
# Run NanoBEIR benchmarks
cd nanobeir && make eval

# View leaderboard
make show
# Expected: R@1, R@5, NDCG@10 scores
```

### 4. Grade Verification:
After merging, verify these for **7/10 (Satisfactory)**:
- [ ] Benchmarking pipeline runs: `make eval` works
- [ ] SigLIP served via Modal: <600ms p95
- [ ] Query2Dish dataset: 3k+ curated pairs
- [ ] First custom model training infrastructure ready

---

## Potential Merge Conflicts

Based on branch analysis, likely conflicts in:
1. `README.md` - Multiple branches modify it
2. `esci-dataset/` directory structure - Overlap between annotation branches
3. `pyproject.toml` - Different dependency versions

**Resolution strategy:** Prefer newer/more complete versions, combine dependencies.
