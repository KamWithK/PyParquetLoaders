from pathlib import Path
from timeit import default_timer
from PyTorchLoader import IterableParquetDataset, IterableManualParquetDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from petastorm import make_batch_reader
from petastorm.transform import TransformSpec
from PetastormDataLoader import TransformersDataLoader

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", fast=True)
def process_rows(columns_dict):
    tokens = tokenizer(columns_dict["readme"], padding=True, truncation=True, return_tensors="pt").data
    columns_dict.update(tokens)
    
    # Remove unwanted columns
    [columns_dict.pop(column) for column in dict(columns_dict) if column not in ["token_type_ids", "attention_mask", "input_ids", "target"]]
    return columns_dict

start = default_timer()
dataset = IterableParquetDataset("Data/Train.parquet", process_rows)
dataloader = DataLoader(dataset, num_workers=4)
list(dataloader)
end = default_timer()
print(end - start)

del dataset, dataloader
del start, end

start = default_timer()
dataset = IterableManualParquetDataset("Data/Train.parquet", process_rows)
dataloader = DataLoader(dataset, num_workers=4)
list(dataloader)
end = default_timer()
print(end - start)

del dataset, dataloader
del start, end

start = default_timer()
dataset = make_batch_reader(Path("Data/Train.parquet").absolute().as_uri(), workers_count=4, transform_spec=TransformSpec(lambda rows: rows[["readme", "target"]], selected_fields=["readme", "target"]))
dataloader = TransformersDataLoader(dataset, string_column="readme")
list(dataloader)
end = default_timer()
print(end - start)