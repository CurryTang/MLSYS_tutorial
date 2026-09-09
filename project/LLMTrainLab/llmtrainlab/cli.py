"""llmctl: walk the local LLM training control plane from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from llmtrainlab.engine import ClusterEngine, default_cluster_spec
from llmtrainlab.store import load_state, save_state, state_path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _load_engine(root: Path | None = None) -> ClusterEngine:
    return ClusterEngine.from_snapshot(load_state(root))


def _save(engine: ClusterEngine, root: Path | None = None) -> None:
    save_state(engine.snapshot(), root)


def _print_table(rows: list[dict[str, Any]], keys: list[str]) -> None:
    if not rows:
        print("(empty)")
        return
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in keys}
    print("  ".join(key.ljust(widths[key]) for key in keys))
    print("  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key in keys))


def cmd_cluster_init(args: argparse.Namespace) -> int:
    spec = default_cluster_spec()
    if args.config:
        spec = yaml.safe_load(Path(args.config).read_text())
    engine = ClusterEngine(spec)
    _save(engine, args.root)
    inv = engine.gpu_inventory()
    print(f"cluster initialized → {state_path(args.root)}")
    print("GPU inventory:")
    for gpu_type, item in inv.items():
        print(f"  {gpu_type}: allocatable={item['allocatable']} total={item['total']}")
    print("nodes:", ", ".join(engine.nodes))
    print("teams:", ", ".join(engine.teams))
    return 0


def cmd_cluster_status(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    print(f"tick={engine.tick}")
    print("nodes:")
    _print_table(
        [
            {
                "name": node.name,
                "type": node.gpu_type,
                "gpus": node.gpus,
                "status": node.status,
                "rack": node.rack,
                "net": node.network,
                "kind": node.kind,
            }
            for node in engine.nodes.values()
        ],
        ["name", "type", "gpus", "status", "rack", "net", "kind"],
    )
    print("\njobs:")
    cmd_job_list(args)
    return 0


def cmd_job_submit(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    raw = yaml.safe_load(Path(args.file).read_text())
    job = engine.submit_job(raw)
    if not args.no_tick:
        engine.step(1)
    _save(engine, args.root)
    print(f"submitted {job.name} phase={engine.jobs[job.name].phase} demand={job.gpu_demand}x{job.gpu_type}")
    return 0


def cmd_job_list(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    rows = []
    for job in engine.jobs.values():
        ckpt = engine._latest_ckpt(job.name)
        live = [w for w in engine.workers.values() if w.job == job.name and w.phase not in {"Stopped"}]
        step = max((w.step for w in live), default=(ckpt.step if ckpt else 0))
        rows.append(
            {
                "name": job.name,
                "ns": job.namespace,
                "phase": job.phase,
                "pri": job.priority,
                "gpus": job.gpu_demand,
                "type": job.gpu_type,
                "step": step,
                "ckpt": ckpt.step if ckpt else "-",
                "retries": job.retries,
            }
        )
    _print_table(rows, ["name", "ns", "phase", "pri", "gpus", "type", "step", "ckpt", "retries"])
    return 0


def cmd_job_status(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    job = engine.jobs[args.name]
    print(json.dumps({**engine.metrics()["jobs"][job.name], "events": job.events[-12:]}, indent=2))
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    engine.step(args.steps)
    _save(engine, args.root)
    print(f"tick={engine.tick}")
    cmd_job_list(args)
    return 0


def cmd_fault(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    kind = args.kind
    if kind == "pod":
        engine.fault_pod(args.target)
    elif kind == "node":
        engine.fault_node(args.target)
    elif kind == "network":
        if not args.job:
            raise SystemExit("fault network 需要 --job")
        engine.fault_network(args.job)
    elif kind == "ckpt":
        engine.fault_checkpoint_store(args.duration)
    elif kind == "gpu":
        engine.shrink_gpus(args.target, args.remaining)
    else:
        raise SystemExit(f"unknown fault {kind}")
    engine.step(args.steps)
    _save(engine, args.root)
    print(f"fault {kind} applied, tick={engine.tick}")
    cmd_job_list(args)
    return 0


def cmd_recover_node(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    engine.recover_node(args.name)
    engine.step(1)
    _save(engine, args.root)
    print(f"node {args.name} Ready, tick={engine.tick}")
    return 0


def cmd_scale(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    created = engine.scale_virtual_nodes(args.nodes, gpus=args.gpus, gpu_type=args.gpu_type)
    engine.step(1)
    _save(engine, args.root)
    print(f"added {created} KWOK-style virtual nodes, total nodes={len(engine.nodes)}")
    print(engine.gpu_inventory())
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    engine = _load_engine(args.root)
    metrics = engine.metrics()
    if args.json:
        print(json.dumps(metrics, indent=2))
        return 0
    print(f"tick={metrics['tick']}  gpu_util={metrics['gpu_slot_util']}  pending_pods={metrics['pending_pods']}")
    print(f"queue wait   p50={metrics['queue_wait_p50']}  p95={metrics['queue_wait_p95']}")
    print(f"admit        p50={metrics['admit_latency_p50']}  p95={metrics['admit_latency_p95']}")
    print(f"recovery     p50={metrics['recovery_p50']}  p95={metrics['recovery_p95']}")
    print(f"throughput   {metrics['training_throughput']} steps/tick")
    print("tenant GPUs:", metrics["tenant_gpus"])
    print("inventory:", metrics["gpu_inventory"])
    return 0


def _until(engine: ClusterEngine, predicate, limit: int = 800) -> None:
    for _ in range(limit):
        if predicate():
            return
        engine.step(1)
    raise RuntimeError(f"timeout at tick={engine.tick}")


def cmd_demo_canonical(args: argparse.Namespace) -> int:
    engine = ClusterEngine(default_cluster_spec())
    print("=== 1. 提交 Job A：6 x H100，低优先级 ===")
    engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "job-a.yaml").read_text()))
    _until(engine, lambda: engine.jobs["llama-a"].phase == "Running")
    print(f"    A running at t={engine.tick}, H100 used={engine.gpu_inventory()['H100']['used']}")

    print("=== 2. 提交 Job B：2 x H100，高优先级（占满剩余 slot）===")
    engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "job-b.yaml").read_text()))
    _until(engine, lambda: engine.jobs["llama-b"].phase == "Running")
    print(f"    B running at t={engine.tick}, H100 used={engine.gpu_inventory()['H100']['used']}")

    print("=== 3. 提交 Job C：4 x H100 → 进入队列 ===")
    engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "job-c.yaml").read_text()))
    engine.step(3)
    print(f"    C phase={engine.jobs['llama-c'].phase} (期望 Queued)")

    print("=== 4. 杀掉 A 的 worker-2，整组从 checkpoint 恢复 ===")
    engine.step(engine.jobs["llama-a"].checkpoint_every + 5)
    ckpt_before = engine._latest_ckpt("llama-a")
    print(f"    checkpoint before fault: step={ckpt_before.step if ckpt_before else None}")
    engine.fault_pod("llama-a-worker-2")
    _until(engine, lambda: engine.jobs["llama-a"].phase == "Running" and engine.jobs["llama-a"].retries >= 1)
    print(
        f"    A recovered retries={engine.jobs['llama-a'].retries} "
        f"lost_steps={engine.jobs['llama-a'].lost_steps} "
        f"ckpt={engine._latest_ckpt('llama-a').step if engine._latest_ckpt('llama-a') else None}"
    )

    print("=== 5. 可选抢占：再提交 8 x H100 的高优先级 inference ===")
    if args.preempt:
        engine.submit_job(yaml.safe_load((EXAMPLES / "jobs" / "preempt-high.yaml").read_text()))
        engine.step(4)
        print(f"    urgent={engine.jobs['llama-urgent'].phase} A={engine.jobs['llama-a'].phase} C={engine.jobs['llama-c'].phase}")

    print("=== 6. KWOK 风格扩到虚拟节点，观察队列被吸干 ===")
    engine.scale_virtual_nodes(args.virtual_nodes, gpus=8, gpu_type="H100")
    engine.step(8)
    print(f"    nodes={len(engine.nodes)} C={engine.jobs['llama-c'].phase}")

    print("=== 7. p50 / p95 ===")
    metrics = engine.metrics()
    print(json.dumps({k: metrics[k] for k in [
        "tick", "gpu_slot_util", "pending_pods", "queue_wait_p50", "queue_wait_p95",
        "admit_latency_p50", "admit_latency_p95", "recovery_p50", "recovery_p95",
        "tenant_gpus",
    ]}, indent=2))
    _save(engine, args.root)
    print(f"\nstate saved to {state_path(args.root)}")
    return 0


def cmd_demo_walk(args: argparse.Namespace) -> int:
    """Print the exact command sequence of the Chinese README."""
    print(
        """
# LLMTrainLab 逐步命令（把每一段贴进终端）

cd project/LLMTrainLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

llmctl cluster init --config examples/cluster.yaml
llmctl cluster status

llmctl job submit examples/jobs/job-a.yaml
llmctl tick --steps 8
llmctl job list

llmctl job submit examples/jobs/job-b.yaml
llmctl tick --steps 4
llmctl job list

llmctl job submit examples/jobs/job-c.yaml
llmctl tick --steps 3
llmctl job list          # C 应仍是 Queued

llmctl tick --steps 120  # 让 A 写出第一个 checkpoint
llmctl fault pod llama-a-worker-2
llmctl job status llama-a
llmctl tick --steps 6
llmctl job status llama-a   # Recovering → Running，retries>=1

llmctl job submit examples/jobs/preempt-high.yaml
llmctl tick --steps 5
llmctl job list             # 低优先级任务被整组抢占

llmctl scale --nodes 40
llmctl tick --steps 10
llmctl metrics
""".strip()
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmctl", description="LLMTrainLab 本地训练控制面")
    parser.add_argument("--root", type=Path, default=None, help="状态目录的父路径，默认 cwd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cluster = sub.add_parser("cluster")
    cluster_sub = cluster.add_subparsers(dest="cluster_cmd", required=True)
    init = cluster_sub.add_parser("init")
    init.add_argument("--config", type=str, default=None)
    init.set_defaults(func=cmd_cluster_init)
    status = cluster_sub.add_parser("status")
    status.set_defaults(func=cmd_cluster_status)

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_cmd", required=True)
    submit = job_sub.add_parser("submit")
    submit.add_argument("file")
    submit.add_argument("--no-tick", action="store_true")
    submit.set_defaults(func=cmd_job_submit)
    listed = job_sub.add_parser("list")
    listed.set_defaults(func=cmd_job_list)
    st = job_sub.add_parser("status")
    st.add_argument("name")
    st.set_defaults(func=cmd_job_status)

    tick = sub.add_parser("tick")
    tick.add_argument("--steps", "-n", type=int, default=1)
    tick.set_defaults(func=cmd_tick)

    fault = sub.add_parser("fault")
    fault.add_argument("kind", choices=["pod", "node", "network", "ckpt", "gpu"])
    fault.add_argument("target", nargs="?", default="")
    fault.add_argument("--job", default="")
    fault.add_argument("--duration", type=int, default=20)
    fault.add_argument("--remaining", type=int, default=0)
    fault.add_argument("--steps", type=int, default=2)
    fault.set_defaults(func=cmd_fault)

    rec = sub.add_parser("recover-node")
    rec.add_argument("name")
    rec.set_defaults(func=cmd_recover_node)

    scale = sub.add_parser("scale")
    scale.add_argument("--nodes", type=int, required=True)
    scale.add_argument("--gpus", type=int, default=8)
    scale.add_argument("--gpu-type", default="H100")
    scale.set_defaults(func=cmd_scale)

    metrics = sub.add_parser("metrics")
    metrics.add_argument("--json", action="store_true")
    metrics.set_defaults(func=cmd_metrics)

    demo = sub.add_parser("demo")
    demo_sub = demo.add_subparsers(dest="demo_cmd", required=True)
    canonical = demo_sub.add_parser("canonical")
    canonical.add_argument("--virtual-nodes", type=int, default=500)
    canonical.add_argument("--preempt", action="store_true")
    canonical.set_defaults(func=cmd_demo_canonical)
    walk = demo_sub.add_parser("walk")
    walk.set_defaults(func=cmd_demo_walk)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"unknown object: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
