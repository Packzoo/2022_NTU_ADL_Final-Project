import torch
import pandas as pd
import numpy as np
import os
import copy
import datetime
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
pd.set_option('mode.chained_assignment', None)
torch.manual_seed(2022)

from sklearn.preprocessing import LabelEncoder

# torch_rehub
from torch_rechub.utils.match import gen_model_input
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import df_to_dict
from model import MLP
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

# own package
from set_arg import set_arg
from utils import match_evaluation, pad_sequences, df_to_dict
import tqdm
from sklearn.metrics import accuracy_score


def train_summary(args):
    
    input_dir = args.input_dir
    train_summary = {"user_id":[], "subgroup":[]}
    train = pd.read_csv(args.train_data)

    ## split subgroup
    for i, cids in enumerate(train["subgroup"]):
        if not pd.isna(cids):
            ids = cids.split(" ")
            for idx in ids: 
                train_summary["user_id"].append(train["user_id"][i])
                train_summary["subgroup"].append(str(int(idx)-1))
        else:
            train_summary["user_id"].append(train["user_id"][i])
            train_summary["subgroup"].append("50")

    train = pd.DataFrame(train_summary)
    
    # add user features
    user = pd.read_csv(input_dir + "users.csv")
    train = train.merge(user, on=["user_id"])

    return train

def eval_summary(args):
    
    input_dir = args.input_dir
    val_summary = {"user_id":[], "subgroup":[]}
    val = pd.read_csv(args.valid_data)

    ## split subgroup
    for i,cids in enumerate(val["subgroup"]):
        if not pd.isna(cids):
            ids = cids.split(" ")
            for idx in ids: 
                val_summary["user_id"].append(val["user_id"][i])
                val_summary["subgroup"].append(str(int(idx)-1))
        else:
            val_summary["user_id"].append(val["user_id"][i])
            val_summary["subgroup"].append("50")

    val = pd.DataFrame(val_summary)

    # add user features
    user = pd.read_csv(input_dir + "users.csv")
    val = val.merge(user, on=["user_id"])

    return val

def test_summary(args):
    
    input_dir = args.input_dir
    test_summary = {"user_id":[],"subgroup":[]}
    test = pd.read_csv(args.test_data)

    ## split subgroup
    for i,cids in enumerate(test["subgroup"]):
        test_summary["user_id"].append(test["user_id"][i])
        test_summary["subgroup"].append("0")

    test = pd.DataFrame(test_summary)

    # add user features
    user = pd.read_csv(input_dir + "users.csv")
    test = test.merge(user,on=["user_id"])

    return test

def summary_csv_to_pd(args):

    train_df = train_summary(args)
    eval_df = eval_summary(args)
    test_df = test_summary(args)

    return train_df, eval_df, test_df



def preprocess(args, train_df, eval_df, test_df):

    user_col = "user_id"
    item_col = "subgroup"

    user_cols = [ "user_id", "gender", "occupation_titles", "interests", "recreation_names"]
    
    # 原本的 users
    train_user = train_df.drop_duplicates('user_id')["user_id"]
    test_user = test_df.drop_duplicates('user_id')["user_id"]

    # split 的 labels
    train_df_label = train_df["subgroup"]
    test_df_label = test_df["subgroup"]

    # split 的 users
    train_df_user = train_df["user_id"]
    test_df_user = test_df["user_id"]

    for feature in user_cols:
        lbe = LabelEncoder()
        lbe.fit(train_df[feature])
        train_df[feature] = lbe.transform(train_df[feature]) + 1
        test_df[feature] = lbe.transform(test_df[feature]) + 1

    df_train = train_df[user_cols]
    df_test = test_df[user_cols]

    # dataframe -> numpy -> tensor
    df_train_array_x = np.array(df_train)
    df_train_tensor_x = torch.from_numpy(df_train_array_x.astype(np.float32))

    df_train_array_y = np.array(train_df_label)
    df_train_tensor_y = torch.from_numpy(df_train_array_y.astype(int))
    
    df_test_array_x = np.array(df_test)
    df_test_tensor_x = torch.from_numpy(df_test_array_x.astype(np.float32))
    
    df_test_array_y = np.array(test_df_label)
    df_test_tensor_y = torch.from_numpy(df_test_array_y.astype(int))


    train_data = torch.utils.data.TensorDataset(df_train_tensor_x, df_train_tensor_y)
    test_data = torch.utils.data.TensorDataset(df_test_tensor_x, df_test_tensor_y)

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_data,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 4,
    )

    return train_dataloader, test_dataloader, test_user, test_df_user

def apk(actual, predicted, k=50):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=50):
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)]) 

def train(args, train_dataloader, test_dataloader, test_user, test_user_split):

    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load(args.save_dir + "model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # test
    results_users = []
    result_preds = []
    for user in test_user_split:
        results_users.append(user)
    
    model.eval()
    for idx, (data, label) in enumerate(tqdm.tqdm(test_dataloader)):
        outputs = model(data)
        for output in outputs:
            result_preds.append(output)
    
    submit_prob = {}
    for idx, user in enumerate(results_users):
        if user in submit_prob:
            submit_prob[user] = submit_prob[user] + result_preds[idx]
        else:
            submit_prob[user] = result_preds[idx]
    
    submit = {"user_id": [], "subgroup": []}
    for user in test_user:
        submit["user_id"].append(user)
    
    for user in submit["user_id"]:
        outputs_prob, outputs_idx = torch.sort(submit_prob[user], descending = True)

        pred_idx_list = outputs_idx[:args.topk].numpy()
        
        pred_str_list = []
        for j in pred_idx_list:
            pred_str_list.append(str(j+1))
        pred_str = " ".join(pred_str_list)

        submit["subgroup"].append(pred_str)

    submit = pd.DataFrame(submit)
    submit.to_csv(args.output_data, index=False)
    
    print(submit)
        

def run(args):

    train_df, eval_df, test_df = summary_csv_to_pd(args)

    train_dataloader, test_dataloader, test_user, test_user_split = preprocess(args, train_df, eval_df, test_df)

    train(args, train_dataloader, test_dataloader, test_user, test_user_split)



if __name__ == "__main__":
    args = set_arg()
    run(args)