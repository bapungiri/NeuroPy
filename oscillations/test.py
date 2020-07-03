from dask.distributed import Client, progress
import dask.array as da

client = Client(processes=False, threads_per_worker=4, n_workers=2, memory_limit="4GB")
client
x = da.random.random((10000, 10000), chunks=(1000, 1000))
x

y = x + x.T
z = y[::2, 5000:].mean(axis=1)
z

client.run_on_scheduler(
    lambda dask_scheduler=None: dask_scheduler.close() & sys.exit(0)
)

