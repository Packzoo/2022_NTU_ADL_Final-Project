"""
    util function for movielens data.
"""
import collections
import numpy as np
import pandas as pd
from torch_rechub.utils.match import Annoy
from torch_rechub.basic.metric import topk_metrics
import tqdm
import random
from collections import OrderedDict, Counter
from annoy import AnnoyIndex

def apk(actual, predicted, k=50):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
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
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        reference: https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR/fuxictr
    """
    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr

def df_to_dict(data):
    """
    Convert the DataFrame to a dict type input that the network can accept
    Args:
        data (pd.DataFrame): datasets of type DataFrame
    Returns:
        The converted dict, which can be used directly into the input network
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict



def match_evaluation(args,user_embedding, item_embedding, test_user, all_item, 
                     raw_id_maps, user_col='user_id', item_col='course_id', topk=10):
    print("evaluate embedding matching on test data")
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)

    #for each user of test dataset, get ann search topk result
    print("matching for topk")
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    match_res = collections.defaultdict(dict)  # user id -> predicted item ids
    preds,users = [],[]
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        if len(user_emb.shape)==2:
            #多兴趣召回
            items_idx = []
            items_scores = []
            for i in range(user_emb.shape[0]):
                temp_items_idx, temp_items_scores = annoy.query(v=user_emb[i], n=topk)  # the index of topk match items
                items_idx += temp_items_idx
                items_scores += temp_items_scores
            temp_df = pd.DataFrame()
            temp_df['item'] = items_idx
            temp_df['score'] = items_scores
            temp_df = temp_df.sort_values(by='score', ascending=True)
            temp_df = temp_df.drop_duplicates(subset=['item'], keep='first', inplace=False)
            recall_item_list = temp_df['item'][:topk].values
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][recall_item_list])
        else:
            #普通召回
            items_idx, items_scores = annoy.query(v=user_emb, n=topk)  #the index of topk match items
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])
        
        preds.append(match_res[user_map[user_id]])
        users.append(user_map[user_id])



    #get ground truth
    if args.test:
        submit = {"user_id":[], "course_id":[]}
        for user,pred in zip(users, preds):
            submit["user_id"].append(user)
            pred = " ".join(pred)
            submit["course_id"].append(pred)
        
        submit = pd.DataFrame(submit)
        print(submit)
        submit.to_csv(args.save_dir + "result.csv", index=False)

    else:
        print("generate ground truth")

        data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
        data[user_col] = data[user_col].map(user_map)
        data[item_col] = data[item_col].map(item_map)
        user_pos_item = data.groupby(user_col, sort=False).agg(list).reset_index()
        # ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user id -> ground truth

        labels = []
        for data in user_pos_item[item_col]:
            labels.append(data)
        
        out = mapk(labels, preds)
        print("mapk score:" + str(out))

        submit = {"user_id":[], "course_id":[]}
        for user, pred in zip(users, preds):
            submit["user_id"].append(user)
            pred = " ".join(pred)
            submit["course_id"].append(pred)
        
        submit = pd.DataFrame(submit)
        print(submit)
        submit.to_csv(args.save_dir + "result_train.csv", index=False)


def get_item_sample_weight(items):
    #It is a effective weight used in word2vec
    items_cnt = Counter(items)
    p_sample = {item: count**0.75 for item, count in items_cnt.items()}
    p_sum = sum([v for k, v in p_sample.items()])
    item_sample_weight = {k: v / p_sum for k, v in p_sample.items()}
    return item_sample_weight
