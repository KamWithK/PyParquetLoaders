import numpy as np
# import pandas as pd

from PetastormDataLoaders.PetastormDataModule import PetastormDataModule
from PetastormDataLoaders.PetastormDataLoader import TransformersDataLoader
# from petastorm.pytorch import DataLoader
from petastorm.transform import TransformSpec
from transformers import AutoTokenizer

class TextParseDataModule(PetastormDataModule):
    def __init__(self, train_path, val_path, batch_size=16, num_minibatchs=8, num_workers=4, text_column="", target_column="target"):
        # Tokenised columns replace text based ones
        remove_other = lambda rows: rows[[text_column, target_column]]
        transform_spec = TransformSpec(remove_other, selected_fields=[text_column, target_column])

        super().__init__(train_path, val_path, batch_size, num_minibatchs, num_workers, transform_spec)

    def train_dataloader(self):
        return TransformersDataLoader(self.train_dataset, string_column="readme")

    def val_dataloader(self):
        return TransformersDataLoader(self.val_dataset, string_column="readme")

# class TextParseDataModule(PetastormDataModule):
#     def __init__(self, train_path, val_path, batch_size=16, num_minibatchs=8, num_workers=4, text_column="", target_column="target"):
#         # Tokenised columns replace text based ones
#         tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", fast=True)
#         tokenize = lambda rows: pd.concat((rows[[target_column]], pd.DataFrame([tokenizer(rows[text_column].tolist(), padding=True, truncation=True, return_tensors="np").data])))

#         edit_fields = [("token_type_ids", np.array, (), False), ("attention_mask", np.array, (), False), ("input_ids", np.array, (), False)]
#         transform_spec = TransformSpec(tokenize, selected_fields=["token_type_ids", "attention_mask", "input_ids", target_column], edit_fields=edit_fields)

#         super().__init__(train_path, val_path, batch_size, num_minibatchs, num_workers, transform_spec)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset)
