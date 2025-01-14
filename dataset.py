from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch


class WritingStyleDataset(Dataset):
    def __init__(
        self,
        dataframe,
        user_column,
        user_id,
        text_column,
        model_name
    ):
        self.df = dataframe
        self.user_id = user_id
        self.user_column = user_column
        self.text_column = text_column
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # attach user written label
        self.df["label"] = self.df[user_column].apply(lambda x: 1 if x == self.user_id else 0)
        # construct a balanced dataframe having roughly same number of user vs non-user text
        user_df_only = self.df[self.df[self.user_column] == self.user_id]
        non_user_df = self.df[self.df[self.user_column] != self.user_id]
        shuffled_non_user_df = non_user_df.sample(len(user_df_only), random_state=42)
        self.balanced_df = pd.concat([user_df_only, shuffled_non_user_df])

    def __len__(self):
        return len(self.balanced_df)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.balanced_df.iloc[idx][self.text_column],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        label = torch.tensor([1]) if self.df.iloc[idx]["label"] == 1 else torch.tensor([0])

        return tokenized, label
