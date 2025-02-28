from utils import load_data
from models.lm_verifier import Verifier
from dataset import WritingStyleDataset
import argparse
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def main(args):
    data_name = args.data_name
    split = args.split
    user_id = args.id
    user_col = args.user_column
    text_col = args.text_column   
    model_name = args.model_name
    # load the dataframe
    df = load_data(data_name, split)
    # convert to torch dataset
    dataset = WritingStyleDataset(
        df,
        user_col,
        user_id,
        text_col,
        model_name
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = Verifier(model_name)
    learning_rate = 1e-5
    batch_size = 8
    num_epochs = 3
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for _, batch in enumerate(tqdm(dataloader)):
            tokenized, labels = batch
            labels = labels.to(device)
            input_ids = tokenized["input_ids"].squeeze(1).to(device)
            attention_mask = tokenized["attention_mask"].squeeze(1).to(device)
            #labels = labels.squeeze
            y_pred = model(input_ids, attention_mask)
            
            optimizer.zero_grad()
            loss = loss_fn(y_pred, labels.float())
            epoch_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()

        
        print(f"EPOCH LOSS: {epoch_loss}")
            
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--data_name", type=str, required=True, help="dataset name: IMDb, Blog, Yelp")
    parser.add_argument("-s", "--split", type=str, required=True, help="data split: train, test, validation")
    parser.add_argument("-txt", "--text_column", type=str, required=True, help="column name which contains user text")
    parser.add_argument("-uc", "--user_column", type=str, required=True, help="column name which contains user id")
    parser.add_argument("-id", type=str, required=True, help="user_id to train a verifier for")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="model name for tokenizing the texts")

    args = parser.parse_args()
    main(args)


