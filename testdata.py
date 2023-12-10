import zipfile
from torch.utils.data import Dataset
import torch
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import requests
import sklearn
import random
from torch.utils.data import DataLoader
from evaluation import Evaluation
from bpr_loss import BPR_Loss


class MovieLens(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 ng_ratio: int,
                 train:bool=False,
                 )->None:
        '''
        :param root: dir for download and train,test.
        :param file_size: large of small. if size if large then it will load(download) 20M dataset. if small then, it will load(download) 100K dataset.
        :param download: if true, it will down load from url.
        '''
        super(MovieLens, self).__init__()

        self.df = df
        self.total_df = total_df
        self.train = train
        self.ng_ratio = ng_ratio
        self.users, self.items = self._negative_sampling()
        print(f'len items:{self.items.shape}')

    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''

        # self.items[index][0]: positive feedback
        # self.items[index][1]: negative feedback
        if self.train:
            return self.users[index], self.items[index][0], self.items[index][1]
        else:
            return self.users[index], self.items[index]


    def _negative_sampling(self) :
        '''
        sampling one positive feedback per one negative feedback
        :return: dataframe
        '''
        df = self.df
        total_df = self.total_df
        users, items = [], []
        user_item_set = set(zip(df['userId'], df['movieId']))
        total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
        all_movieIds = total_df['movieId'].unique()
        # negative feedback dataset ratio
        for u, i in user_item_set:
            # positive instance
            visit = []
            item = []
            if not self.train:
                items.append(i)
                users.append(u)
            else:
                item.append(i)

            for k in range(self.ng_ratio):
                # negative instance
                negative_item = np.random.choice(all_movieIds)
                # check if item and user has interaction, if true then set new value from random
                while (u, negative_item) in total_user_item_set or negative_item in visit:
                    negative_item = np.random.choice(all_movieIds)

                if self.train:
                    item.append(negative_item)
                    visit.append(negative_item)
                else:
                    items.append(negative_item)
                    visit.append(negative_item)
                    users.append(u)

            if self.train:
                items.append(item)
                users.append(u)

        return torch.tensor(users), torch.tensor(items)

def _read_ratings_csv(fname) -> pd.DataFrame:
    '''
    at first, check if file exists. if it doesn't then call _download().
    it will read ratings.csv, and transform to dataframe.
    it will drop columns=['timestamp'].
    :return:
    '''
    print("Reading file")

    df = pd.read_csv(fname, sep="::", header=None,
                        names=['userId', 'movieId', 'ratings', 'timestamp'])
    df = df.drop(columns=['timestamp'])
    print("Reading Complete!")
    return df

def split_train_test(df) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    pick each unique userid row, and add to the testset, delete from trainset.
    :return: (pd.DataFrame,pd.DataFrame,pd.DataFrame)
    '''
    train_dataframe = df
    test_dataframe = df.sample(frac=1).drop_duplicates(['userId'])
    tmp_dataframe = pd.concat([train_dataframe, test_dataframe])
    train_dataframe = tmp_dataframe.drop_duplicates(keep=False)

    # explicit feedback -> implicit feedback
    # ignore warnings
    np.warnings.filterwarnings('ignore')
    train_dataframe.loc[:, 'rating'] = 1
    test_dataframe.loc[:, 'rating'] = 1

    print(f"len(total): {len(df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
    return df, train_dataframe, test_dataframe,

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

fname=os.path.join('..\\NGCF-master\\resource\\ml-1m', 'ratings.dat')
df=_read_ratings_csv(fname)
total_df , train_df, test_df = split_train_test(df)

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print GPU information
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


# make torch.utils.data.Data object
test_set = MovieLens(df=test_df,total_df=total_df,train=False,ng_ratio=99)


load_model = torch.load("pretrain\\NGCF.pth")[2]
load_model.cuda()

test_loader = DataLoader(test_set,
                         batch_size=100,
                         shuffle=False,
                         drop_last=True
                         )


criterion = BPR_Loss(batch_size=256,decay_ratio=1e-5)
avg_cost = 0
total_batch = len(test_loader)
for idx,(users,pos_items,neg_items) in enumerate(test_loader):
    users,pos_items,neg_items = users.to(device),pos_items.to(device),neg_items.to(device)
    user_embeddings, pos_item_embeddings, neg_item_embeddings= load_model(users,pos_items,neg_items,use_dropout=True)
    cost  = criterion(user_embeddings,pos_item_embeddings,neg_item_embeddings)
    avg_cost+=cost
avg_cost = avg_cost/total_batch
eval = Evaluation(test_dataloader=test_loader,
                    model = load_model,
                    top_k=10,
                    device=device)
HR,NDCG = eval.get_metric()
print("HR: {:.3f}\tNDCG: {:.3f}".format(HR, NDCG))