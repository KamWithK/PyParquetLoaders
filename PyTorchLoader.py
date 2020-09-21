import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd

from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from torch.multiprocessing import Queue

class IterableManualParquetDataset(IterableDataset):
    def __init__(self, path, process_func):
        super().__init__()
        self.dataset = ds.dataset(path)
        self.process_func = process_func

    def __iter__(self):
        worker_info = get_worker_info()

        # Only divide up batches when using multiple worker processes
        if worker_info != None:
            batches = list(self.dataset.to_batches())
            worker_load = len(batches) // worker_info.num_workers

            # If more workers than batches exist, some won't be used
            if worker_load == 0:
                if worker_info.id < len(batches): self.batches = [batches[worker_info.id]]
                else: return
            else:
                start = worker_load * worker_info.id
                end = min(start + worker_load, len(batches))
                self.batches = batches[start:end]
        else: self.batches = self.dataset.to_batches()

        # Process and yield each batch
        for batch in self.batches:
            batch = batch.to_pydict()
            batch.update(self.process_func(batch))

            yield batch

class IterableParquetDataset(IterableDataset):
    def __init__(self, path, process_func):
        super().__init__()
        dataset = ds.dataset(path)
        self.process_func = process_func

        self.batches = Queue()
        [self.batches.put(batch) for batch in dataset.to_batches()]

    def __iter__(self):
        while True:
            if self.batches.empty() == True:
                self.batches.close()
                break

            batch = self.batches.get().to_pydict()
            batch.update(self.process_func(batch))
            yield batch
