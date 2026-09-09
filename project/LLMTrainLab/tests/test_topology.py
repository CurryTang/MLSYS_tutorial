from llmtrainlab.engine import ClusterEngine, default_cluster_spec


def test_h100_job_does_not_land_on_a100():
    engine = ClusterEngine(default_cluster_spec())
    engine.submit_job(
        {
            "metadata": {"name": "need-h100", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "priority": 50,
                "workers": 2,
                "gpusPerWorker": 1,
                "gpuType": "H100",
                "steps": 10,
                "checkpointEvery": 5,
                "requireRdma": True,
            },
        }
    )
    engine.step(4)
    nodes = {w.node for w in engine.workers.values() if w.job == "need-h100"}
    assert nodes == {"node-b"}
    assert all(engine.nodes[name].gpu_type == "H100" for name in nodes)


def test_rdma_job_skips_ethernet_nodes():
    engine = ClusterEngine(default_cluster_spec())
    engine.submit_job(
        {
            "metadata": {"name": "need-rdma", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "priority": 50,
                "workers": 2,
                "gpusPerWorker": 1,
                "gpuType": "A100",
                "steps": 10,
                "checkpointEvery": 5,
                "requireRdma": True,
            },
        }
    )
    engine.step(5)
    assert engine.jobs["need-rdma"].phase == "Queued"
    assert engine.gpu_inventory()["A100"]["used"] == 0


def test_same_rack_preferred_when_capacity_allows():
    engine = ClusterEngine(
        {
            "nodes": [
                {"name": "r1a", "gpu_type": "H100", "gpus": 4, "rack": "rack-1", "network": "rdma"},
                {"name": "r2a", "gpu_type": "H100", "gpus": 4, "rack": "rack-2", "network": "rdma"},
            ],
            "teams": {"team-a": {"namespace": "team-a", "guaranteed_gpus": 8, "burst_gpus": 8, "priority": 50}},
        }
    )
    engine.submit_job(
        {
            "metadata": {"name": "packed", "namespace": "team-a"},
            "spec": {
                "team": "team-a",
                "priority": 50,
                "workers": 4,
                "gpusPerWorker": 1,
                "gpuType": "H100",
                "steps": 10,
                "checkpointEvery": 5,
                "requireRdma": True,
                "preferSameRack": True,
            },
        }
    )
    engine.step(3)
    racks = {engine.nodes[w.node].rack for w in engine.workers.values() if w.job == "packed"}
    assert len(racks) == 1
