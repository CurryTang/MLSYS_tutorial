from llmtrainlab.engine import ClusterEngine


def test_worker_kill_restarts_group_from_checkpoint():
    engine = ClusterEngine(
        {
            "nodes": [{"name": "n1", "gpu_type": "H100", "gpus": 4, "network": "rdma"}],
            "teams": {"team-a": {"namespace": "team-a", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 50}},
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "run", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "priority": 50,
                "workers": 4,
                "gpusPerWorker": 1,
                "gpuType": "H100",
                "steps": 400,
                "checkpointEvery": 20,
                "requireRdma": True,
            },
        }
    )
    engine.step(30)
    assert engine.jobs["run"].phase == "Running"
    ckpt = engine._latest_ckpt("run")
    assert ckpt is not None and ckpt.step >= 20

    engine.fault_pod("run-worker-2")
    engine.step(6)
    job = engine.jobs["run"]
    assert job.retries >= 1
    assert job.phase in {"Starting", "Running"}
    assert job.lost_steps <= job.checkpoint_every
    live = [w for w in engine.workers.values() if w.job == "run" and w.phase not in {"Stopped"}]
    assert len(live) == 4
    restored = min(w.step for w in live)
    assert restored >= ckpt.step
    assert restored < ckpt.step + job.checkpoint_every + 10
