from dask.distributed import Client
from dask_jobqueue import SGECluster


def run_jobs(
        fn,
        inputs,
        jobs=1,
        memory='1GB',
        timeout_s=str(3600 * 24 * 7),  # a week
        queue='all.q',
        proc_per_worker=1,
        cores_per_proc=1,
        env_extra=None,
        **kwargs
):
    if env_extra is None:
        env_extra = []
    qsub_mem_str = f'mem_free={memory},ram_free={memory}'.replace('GB', 'G')
    with SGECluster(
            queue=queue,
            walltime=timeout_s,
            processes=proc_per_worker,
            memory=memory,
            cores=cores_per_proc,
            resource_spec=qsub_mem_str,
            env_extra=env_extra  # e.g. ['export ENV_VARIABLE="SOMETHING"', 'source myscript.sh']
    ) as cluster:
        with Client(cluster) as client:
            cluster.scale(jobs)
            futures = client.map(fn, inputs)
            results = client.gather(futures)
    return results
