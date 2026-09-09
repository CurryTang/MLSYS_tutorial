from llmtrainlab.engine import ClusterEngine, default_cluster_spec


def test_gang_does_not_partially_start():
    engine = ClusterEngine(
        {
            "nodes": [{"name": "n1", "gpu_type": "H100", "gpus": 3, "network": "rdma"}],
            "teams": {"team-a": {"namespace": "team-a", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 50}},
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "g4", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "priority": 50,
                "workers": 4,
                "gpusPerWorker": 1,
                "gpuType": "H100",
                "steps": 20,
                "checkpointEvery": 10,
                "requireRdma": True,
            },
        }
    )
    engine.step(5)
    assert engine.jobs["g4"].phase == "Queued"
    assert engine.gpu_inventory()["H100"]["used"] == 0

    engine.add_node({"name": "n2", "gpu_type": "H100", "gpus": 1, "network": "rdma"})
    engine.step(4)
    assert engine.jobs["g4"].phase in {"Starting", "Running"}
    live = [w for w in engine.workers.values() if w.job == "g4" and w.phase not in {"Stopped", "Failed"}]
    assert len(live) == 4
    assert engine.gpu_inventory()["H100"]["used"] == 4
