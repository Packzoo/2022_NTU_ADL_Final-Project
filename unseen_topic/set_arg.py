import pandas as pd
import os
import datetime
import numpy as np
import json
import argparse

def set_arg():
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--input_dir", type=str,default="./input/hahow")
    parser.add_argument("--load_cache", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # model
    parser.add_argument('--model_name', default='dssm')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')  #cuda:0
    parser.add_argument('--save_dir', default='./output/')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--test', action="store_true")

    # data    
    parser.add_argument('--test_data', default='./output/')
    parser.add_argument('--valid_data', default='./output/')
    parser.add_argument('--train_data', default='./output/')
    parser.add_argument('--output_data', default='./output/')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--frequency', type=int, default=0)

    
    args = parser.parse_args()
    return args