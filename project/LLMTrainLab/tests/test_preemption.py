from llmtrainlab.engine import ClusterEngine


def test_high_priority_job_preempts_whole_gang():
    engine = ClusterEngine(
        {
            "nodes": [{"name": "n1", "gpu_type": "H100", "gpus": 8, "network": "rdma"}],
            "teams": {
                "team-a": {"namespace": "team-a", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 10},
                "team-c": {"namespace": "team-c", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 90},
            },
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "low", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "priority": 10,
                "workers": 6,
                "gpusPerWorker": 1,
                "gpuType": "H100",
                "steps": 500,
                "checkpointEvery": 50,
                "requireRdma": True,
            },
        }
    )
    engine.step(4)
    assert engine.jobs["low"].phase == "Running"

    engine.submit_job(
        {
            "metadata": {"name": "high", "namespace": "team-c"},
            "spec": {
                "team": "team-c",
                "priority": 90,
                "workers": 8,
                "gpusPerWorker": 1,
                "gpuType": "H100",
                "steps": 20,
                "checkpointEvery": 10,
                "requireRdma": True,
            },
        }
    )
    engine.step(5)
    assert engine.jobs["high"].phase in {"Starting", "Running"}
    assert engine.jobs["low"].phase == "Preempted"
    assert engine.gpu_inventory()["H100"]["used"] == 8
