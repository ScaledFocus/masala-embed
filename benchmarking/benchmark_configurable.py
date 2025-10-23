import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

DEFAULT_PAYLOADS = "payloads.txt"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
REQUEST_TIMEOUT = 120.0
TICK = 0.1

MODES = {
    "local": {
        "target_url": "http://127.0.0.1:8080/v1/embeddings",
        "concurrency": 100,
        "outfile": "results_local.txt",
        "stages": [(20, 5), (40, 20), (60, 50)],
    },
    "server": {
        "target_url": "https://scaledfocus--masala-embed-server-modal-fastapi-app.modal.run/v1/embeddings",
        "concurrency": 100,
        "outfile": "results_server.txt",
        "stages": [(20, 5), (40, 20), (60, 50)],
    },
}


@dataclass
class Stage:
    duration_s: float
    target_rps: float
    label: str | None = None


@dataclass
class RequestRecord:
    ts: float
    stage_idx: int
    ok: bool
    status: int | None
    latency_ms: float | None
    vllm_ms: float | None
    normalize_ms: float | None
    search_ms: float | None
    server_ms: float | None
    error: str | None


@dataclass
class Summary:
    requests: int
    successes: int
    failures: int
    duration_s: float
    throughput_rps: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    vllm_p50_ms: float | None
    vllm_p95_ms: float | None
    vllm_p99_ms: float | None
    normalize_p50_ms: float | None
    normalize_p95_ms: float | None
    normalize_p99_ms: float | None
    search_p50_ms: float | None
    search_p95_ms: float | None
    search_p99_ms: float | None
    server_p50_ms: float | None
    server_p95_ms: float | None
    server_p99_ms: float | None


def load_payloads(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing payloads file: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise SystemExit("No queries found in payloads file.")
    return lines


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def comp_percentiles(
    vals: list[float],
) -> tuple[float | None, float | None, float | None]:
    if not vals:
        return (None, None, None)
    s = sorted(vals)
    return (percentile(s, 50), percentile(s, 95), percentile(s, 99))


def summarize(records: list[RequestRecord]) -> Summary:
    if not records:
        return Summary(
            requests=0,
            successes=0,
            failures=0,
            duration_s=0.0,
            throughput_rps=0.0,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
            vllm_p50_ms=None,
            vllm_p95_ms=None,
            vllm_p99_ms=None,
            normalize_p50_ms=None,
            normalize_p95_ms=None,
            normalize_p99_ms=None,
            search_p50_ms=None,
            search_p95_ms=None,
            search_p99_ms=None,
            server_p50_ms=None,
            server_p95_ms=None,
            server_p99_ms=None,
        )
    successes = sum(1 for r in records if r.ok)
    failures = len(records) - successes
    min_ts = min(r.ts for r in records)
    max_ts = max(r.ts for r in records)
    duration_s = max(0.0, max_ts - min_ts)
    throughput = (successes / duration_s) if duration_s > 0 else 0.0
    latencies = [r.latency_ms for r in records if r.latency_ms is not None]
    p50, p95, p99 = comp_percentiles(latencies)
    v50, v95, v99 = comp_percentiles(
        [r.vllm_ms for r in records if r.vllm_ms is not None]
    )
    n50, n95, n99 = comp_percentiles(
        [r.normalize_ms for r in records if r.normalize_ms is not None]
    )
    s50, s95, s99 = comp_percentiles(
        [r.search_ms for r in records if r.search_ms is not None]
    )
    sv50, sv95, sv99 = comp_percentiles(
        [r.server_ms for r in records if r.server_ms is not None]
    )
    return Summary(
        requests=len(records),
        successes=successes,
        failures=failures,
        duration_s=duration_s,
        throughput_rps=throughput,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        vllm_p50_ms=v50,
        vllm_p95_ms=v95,
        vllm_p99_ms=v99,
        normalize_p50_ms=n50,
        normalize_p95_ms=n95,
        normalize_p99_ms=n99,
        search_p50_ms=s50,
        search_p95_ms=s95,
        search_p99_ms=s99,
        server_p50_ms=sv50,
        server_p95_ms=sv95,
        server_p99_ms=sv99,
    )


def format_summary_lines(title: str, s: Summary) -> list[str]:
    lines: list[str] = []
    lines.append(title)
    lines.append(
        f"Requests: {s.requests}  Successes: {s.successes}  Failures: {s.failures}"
    )
    lines.append(f"Measured wall-time (first->last): {s.duration_s:.3f} s")
    lines.append(
        f"Throughput (successful reqs / wall-time): {s.throughput_rps:.3f} req/s"
    )
    if s.p50_ms is not None:
        lines.append(
            f"End-to-end latency p50/p95/p99: "
            f"{s.p50_ms:.2f} / {s.p95_ms:.2f} / {s.p99_ms:.2f} ms"
        )

    def add_comp(label: str, p50: float | None, p95: float | None, p99: float | None):
        if p50 is not None:
            lines.append(f"{label} p50/p95/p99: {p50:.2f} / {p95:.2f} / {p99:.2f} ms")

    add_comp("vLLM", s.vllm_p50_ms, s.vllm_p95_ms, s.vllm_p99_ms)
    add_comp("Normalize", s.normalize_p50_ms, s.normalize_p95_ms, s.normalize_p99_ms)
    add_comp("Search", s.search_p50_ms, s.search_p95_ms, s.search_p99_ms)
    add_comp("Server", s.server_p50_ms, s.server_p95_ms, s.server_p99_ms)
    return lines


def parse_stages_arg(stages_arg: str) -> list[Stage]:
    stages: list[Stage] = []
    for part in stages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"Invalid stage spec: {part!r}. Use 'duration:rps'.")
        dur_s, rps = part.split(":", 1)
        try:
            d = float(dur_s.strip())
            r = float(rps.strip())
        except ValueError:
            raise SystemExit(f"Invalid numbers in stage spec: {part!r}")
        if d <= 0 or r <= 0:
            raise SystemExit(f"Stage values must be positive: {part!r}")
        stages.append(Stage(duration_s=d, target_rps=r))
    if not stages:
        raise SystemExit("No valid stages parsed from --stages.")
    return stages


async def do_request(
    session: aiohttp.ClientSession,
    target_url: str,
    model_name: str,
    payload_text: str,
    timeout_s: float,
) -> RequestRecord:
    start = time.perf_counter()
    try:
        async with session.post(
            target_url,
            json={"input": payload_text, "model": model_name},
            timeout=timeout_s,
        ) as resp:
            await resp.read()
            latency_ms = (time.perf_counter() - start) * 1000.0

            def header_f(name: str) -> float | None:
                v = resp.headers.get(name)
                if v is None:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            return RequestRecord(
                ts=time.time(),
                stage_idx=-1,
                ok=(resp.status == 200),
                status=resp.status,
                latency_ms=latency_ms,
                vllm_ms=header_f("X-Timing-vLLM"),
                normalize_ms=header_f("X-Timing-Normalize"),
                search_ms=header_f("X-Timing-Search"),
                server_ms=header_f("X-Timing-Server"),
                error=None,
            )
    except (aiohttp.ClientError, TimeoutError) as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestRecord(
            ts=time.time(),
            stage_idx=-1,
            ok=False,
            status=None,
            latency_ms=latency_ms,
            vllm_ms=None,
            normalize_ms=None,
            search_ms=None,
            server_ms=None,
            error=str(e),
        )


async def run_stage(
    session: aiohttp.ClientSession,
    payloads: list[str],
    stage: Stage,
    target_url: str,
    model_name: str,
    concurrency: int,
    max_requests_holder: dict[str, int | None],
    stage_idx: int,
) -> list[RequestRecord]:
    end_time = time.perf_counter() + stage.duration_s
    per_tick = stage.target_rps * TICK
    remainder = 0.0
    payload_idx = 0

    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task] = []
    records: list[RequestRecord] = []

    async def schedule_one(payload_text: str):
        nonlocal records
        async with sem:
            rem = max_requests_holder.get("remaining", None)
            if rem is not None:
                if rem <= 0:
                    return
                max_requests_holder["remaining"] = rem - 1
            rec = await do_request(
                session=session,
                target_url=target_url,
                model_name=model_name,
                payload_text=payload_text,
                timeout_s=REQUEST_TIMEOUT,
            )
            rec.stage_idx = stage_idx
            records.append(rec)

    while time.perf_counter() < end_time:
        rem = max_requests_holder.get("remaining", None)
        if rem is not None and rem <= 0:
            break
        to_send = per_tick + remainder
        n = int(to_send)
        remainder = to_send - n
        for _ in range(n):
            rem = max_requests_holder.get("remaining", None)
            if rem is not None and rem <= 0:
                break
            payload = payloads[payload_idx % len(payloads)]
            payload_idx += 1
            tasks.append(asyncio.create_task(schedule_one(payload)))
        await asyncio.sleep(TICK)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "server"], default="local")
    p.add_argument("--payloads", default=DEFAULT_PAYLOADS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-requests", type=int, default=None)
    return p


async def main_async(args: argparse.Namespace) -> None:
    cfg = MODES[args.mode]
    target_url: str = cfg["target_url"]
    default_concurrency: int = cfg["concurrency"]
    out_file = args.out or cfg["outfile"]
    payloads = load_payloads(args.payloads)
    concurrency = args.concurrency or default_concurrency
    stages: list[Stage]
    if args.stages:
        stages = parse_stages_arg(args.stages)
    else:
        stages = [Stage(d, r) for (d, r) in cfg["stages"]]

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        max_requests_holder: dict[str, int | None] = {"remaining": args.max_requests}
        all_records: list[RequestRecord] = []
        stage_summaries: list[tuple[Stage, Summary]] = []
        for idx, stage in enumerate(stages):
            print(
                f"Running stage {idx + 1}/{len(stages)}:"
                f"duration={stage.duration_s}s target_rps={stage.target_rps}"
                f"{' label=' + stage.label if stage.label else ''}"
            )
            recs = await run_stage(
                session=session,
                payloads=payloads,
                stage=stage,
                target_url=target_url,
                model_name=args.model,
                concurrency=concurrency,
                max_requests_holder=max_requests_holder,
                stage_idx=idx,
            )
            all_records.extend(recs)
            stage_summaries.append((stage, summarize(recs)))
            rem = max_requests_holder.get("remaining", None)
            if rem is not None and rem <= 0:
                break

        overall = summarize(all_records)

    lines: list[str] = []
    lines.append("=== Benchmark Summary ===")
    lines.append(f"Mode: {args.mode}")
    lines.append(f"Target: {target_url}")
    lines.append(f"Concurrency: {concurrency}")
    lines.append(f"Payloads: {len(payloads)}")
    lines.append(f"Request timeout: {REQUEST_TIMEOUT:.0f}s")
    if args.max_requests is not None:
        lines.append(f"Request cap: {args.max_requests}")
    lines.append("")

    if stage_summaries:
        lines.append("=== Per-stage results ===")
    for idx, (stage, summ) in enumerate(stage_summaries):
        label = stage.label or f"stage-{idx + 1}"
        lines.extend(
            format_summary_lines(
                f"--- {label} (duration={stage.duration_s}s, "
                f"target_rps={stage.target_rps}) ---",
                summ,
            )
        )
        lines.append("")
    lines.extend(format_summary_lines("=== Overall ===", overall))
    lines.append("")
    report = "\n".join(lines)
    print(report)
    Path(out_file).write_text(report, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
