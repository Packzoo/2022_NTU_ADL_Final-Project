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
'''
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
    return 

def gen_model_input(df, user_profile, user_col, item_profile, item_col, seq_max_len, padding='pre', truncating='pre'):
    #merge user_profile and item_profile, pad history seuence feature
    print(df["user_id"])
    df = pd.merge(df, user_profile, on=user_col, how='left')  # how=left to keep samples order same as the input
    df = pd.merge(df, item_profile, on=item_col, how='left')
    print(df["user_id"])

    for col in df.columns.to_list():
        if col.startswith("hist_"):
            df[col] = pad_sequences(df[col], maxlen=seq_max_len, value=0, padding=padding, truncating=truncating).tolist()
    input_dict = df_to_dict(df)
    return input_dict
'''
def apk(actual, predicted, k=10):
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

def negative_sample(items_cnt_order, ratio, method_id=0):
    """Negative Sample method for matching model
    reference: https://github.com/wangzhegeek/DSSM-Lookalike/blob/master/utils.py
    update more method and redesign this function.
    Args:
        items_cnt_order (dict): the item count dict, the keys(item) sorted by value(count) in reverse order.
        ratio (int): negative sample ratio, >= 1
        method_id (int, optional): 
        `{
            0: "random sampling", 
            1: "popularity sampling method used in word2vec", 
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`. 
            Defaults to 0.
            
    Returns:
        list: sampled negative item list
    """
    items_set = [item for item, count in items_cnt_order.items()]
    if method_id == 0:
        neg_items = np.random.choice(items_set, size=ratio, replace=True)
    elif method_id == 1:
        #items_cnt_freq = {item: count/len(items_cnt) for item, count in items_cnt_order.items()}
        #p_sel = {item: np.sqrt(1e-5/items_cnt_freq[item]) for item in items_cnt_order}
        #The most popular paramter is item_cnt**0.75:
        p_sel = {item: count**0.75 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 2:
        p_sel = {item: np.log(count + 1) + 1e-6 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 3:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1) / np.log(len(items_cnt_order) + 1)) for item, k in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=False, p=p_value)
    else:
        raise ValueError("method id should in (0,1,2,3)")
    return neg_items


def generate_seq_feature(train_data,
                                eval_data,
                               user_col,
                               item_col,
                               item_attribute_cols=None,
                               sample_method=0,
                               mode=0,
                               neg_ratio=0,
                               min_item=0):
    """generate sequence feature and negative sample for match.
    Args:
        data (pd.DataFrame): the raw data.
        user_col (str): the col name of user_id 
        item_col (str): the col name of item_id 
        time_col (str): the col name of timestamp
        item_attribute_cols (list[str], optional): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        sample_method (int, optional): the negative sample method `{
            0: "random sampling", 
            1: "popularity sampling method used in word2vec", 
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`. 
            Defaults to 0.
        mode (int, optional): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        neg_ratio (int, optional): negative sample ratio, >= 1. Defaults to 0.
        min_item (int, optional): the min item each user must have. Defaults to 0.
    Returns:
        pd.DataFrame: split train and test data with sequence features.
    """
    if item_attribute_cols is None:
        item_attribute_cols = []
    if mode == 2:  # list wise learning
        assert neg_ratio > 0, 'neg_ratio must be greater than 0 when list-wise learning'
    elif mode == 1:  # pair wise learning
        neg_ratio = 1
    print("preprocess data")
    train_set, test_set = [], []
    n_cold_user = 0

    items_cnt = Counter(train_data[item_col].tolist())
    items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))  #item_id:item count
    neg_list = negative_sample(items_cnt_order, ratio=train_data.shape[0] * neg_ratio, method_id=sample_method)
    neg_idx = 0

    for uid, hist in tqdm.tqdm(train_data.groupby(user_col,sort=False), desc='generate sequence features'):

        pos_list = hist[item_col].tolist()
        if len(pos_list) < min_item:  #drop this user when his pos items < min_item
            n_cold_user += 1
            continue

        for i in range(1, len(pos_list)):

            hist_item = pos_list[:i]
            sample = [uid, pos_list[i], hist_item, len(hist_item)]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:  #the history of item attribute features
                    sample.append(hist[attr_col].tolist()[:i])

            if i != len(pos_list):
                if mode == 0:  #point-wise, the last col is label_col, include label 0 and 1
                    last_col = "label"
                    train_set.append(sample + [1])
                    for _ in range(neg_ratio):
                        sample[1] = neg_list[neg_idx]
                        neg_idx += 1
                        train_set.append(sample + [0])
                elif mode == 1:  #pair-wise, the last col is neg_col, include one negative item
                    last_col = "neg_items"
                    for _ in range(neg_ratio):
                        sample_copy = copy.deepcopy(sample)
                        sample_copy.append(neg_list[neg_idx])
                        neg_idx += 1
                        train_set.append(sample_copy)
                elif mode == 2:  #list-wise, the last col is neg_col, include neg_ratio negative items
                    last_col = "neg_items"
                    sample.append(neg_list[neg_idx: neg_idx + neg_ratio])
                    neg_idx += neg_ratio
                    train_set.append(sample)
                else:
                    raise ValueError("mode should in (0,1,2)")
            else:
                continue  #Note: if mode=1 or 2, the label col is useless.

    for uid, hist in tqdm.tqdm(eval_data.groupby(user_col,sort=False), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        sample = [uid, pos_list[0], [], 0]
        test_set.append(sample + [1])  #Note: if mode=1 or 2, the label col is useless
    
    #random.shuffle(train_set)
    #random.shuffle(test_set)
    
    print("n_train: %d, n_test: %d" % (len(train_set), len(test_set)))
    print("%d cold start user droped " % (n_cold_user))

    attr_hist_col = ["hist_" + col for col in item_attribute_cols]
    df_train = pd.DataFrame(train_set,
                            columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col] + attr_hist_col + [last_col])
    df_test = pd.DataFrame(test_set,
                           columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col] + attr_hist_col + [last_col])

    return df_train, df_test


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

    if not args.test:
        data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
        data[user_col] = data[user_col].map(user_map)
        data[item_col] = data[item_col].map(item_map)
        user_pos_item = data.groupby(user_col,sort=False).agg(list).reset_index()
        
        labels = []
        for data in user_pos_item[item_col]:
            labels.append(data)
        
        mapk_res = mapk(labels, preds)
        print("mapk score:" + str(mapk_res))
        return mapk_res
    else:
        submit = {"user_id":[],"course_id":[]}
        for user,pred in zip(users,preds):
            submit["user_id"].append(user)
            pred = " ".join(pred)
            submit["course_id"].append(pred)
        
        submit = pd.DataFrame(submit)
        submit.to_csv(args.output_data, index=False)
        return 0.0

def get_item_sample_weight(items):
    #It is a effective weight used in word2vec
    items_cnt = Counter(items)
    p_sample = {item: count**0.75 for item, count in items_cnt.items()}
    p_sum = sum([v for k, v in p_sample.items()])
    item_sample_weight = {k: v / p_sum for k, v in p_sample.items()}
    return item_sample_weight
