import os
import json
import numpy as np
import itertools
from collections import Counter
import pickle

from transformers import GenerationConfig, AutoModel, AutoTokenizer
import transformers
import torch
import os
import math
import json
from tqdm import tqdm
import numpy as np 


def load_mat(path):
    data = np.load(path)
    # data = np.random.rand(5, 10)
    return data

def get_item2id(path):
    with open(path) as file:
        datamap = json.load(file)
    return datamap['item2id_dict']

def get_id2item(path):
    with open(path) as file:
        datamap = json.load(file)
    return datamap['id2item_dict']

class MovieENV(object):
    
    def __init__(self, config) -> None:
        path = os.path.join(config.env_path, 'movie/')
        
        self.item2id = get_item2id(os.path.join(path, 'datamaps.json'))
        self.id2item = get_id2item(os.path.join(path, 'datamaps.json'))
        self.reward_mat = load_mat(os.path.join(path, 'movielens_test.npy'))
        print(f'reward_mat shape:{self.reward_mat.shape}')
        # self.reward_mat = np.random.rand(6040, 3952)
        with open(os.path.join(path, 'test_distance_mat.pickle'), 'rb') as f:
            self.distance_mat = pickle.load(f)
        
        # self.distance_mat = np.random.rand(3952, 3952)
        
        with open(os.path.join(path, 'list_genre.json'), 'r') as file:
            self.list_categories = json.load(file)
        
        self.env_window_length = config.env_window_length
        self.threshold = config.env_threshold
        
    def get_reward(self, userid, item):
        
        if item in self.item2id.keys():
            itemid = self.item2id[item]
            reward = self.reward_mat[userid, itemid]
            return reward
        
        else:
            return 0
    
    def whether_to_leave(self, userid, item, item_list):
        # True: Leave.   False: Continue.
        for i, history_item in enumerate(reversed(item_list[:-1])):
            itemid = self.item2id[item]
            history_itemid = self.item2id[history_item]
            distance = self.distance_mat[itemid, history_itemid]
            reward = self.get_reward(userid, item)
            if i < self.env_window_length and (distance < self.threshold or reward < 2):
                print(f"userid:{userid}; item:{item}; history_item:{history_item}; distance:{distance}; reward: {reward}")
                return True
        
        return False
    
    # def whether_to_leave(self, userid, item, item_list):
        
    #     window_items = item_list[-self.env_window_length: ]
    #     hist_categories_each = list(map(lambda x: self.list_categories[x], window_items))
        
    #     hist_categories = list(itertools.chain(*hist_categories_each))
    #     hist_dict = Counter(hist_categories)
    #     category_a = self.list_categories[item]
    #     for c in category_a:
    #         if hist_dict[c] > self.threshold:
    #             print(f"Over recommendation: userid: {userid}, window item:{window_items}, hist_dict:{hist_dict}, the over recommend categories:{c}")
    #             return True
        
    #     return False
        
    def get_item_list(self):
        return self.item2id.keys()
    
    def get_hist_list(self, item_list):
        window_items = item_list[-self.env_window_length: ]
        hist_categories_each = list(map(lambda x: self.list_categories[x], window_items))
        
        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        hist = [key for key, value in hist_dict.items() if value >= 1]
        return hist
    
    

def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]
                
class Movie_Grounding_Model(object):
    
    def __init__(self, model_path, config, batch_size=16):
        super(Movie_Grounding_Model, self).__init__()
        
        self.model_path = model_path
        self.batch_size = batch_size
        
        if 'llama' in self.model_path:
            base_model = self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModel.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.eval()

            self.tokenizer.padding_side = "left"

            self.movie_embedding = torch.load("./env/movie/movie_embedding_task.pt")['embeddings'].cuda()
            self.movie_index = torch.load("./env/movie/movie_embedding_task.pt")['indexs'].cuda()
            
            with open('./env/movie/datamaps.json') as f:
                self.id2item = json.load(f)['id2item_dict']
    
    def get_top1_near_item(self, item_list):
        item_embeddings = []

        for i, batch_input in tqdm(enumerate(batch(item_list, self.batch_size))):
            input = self.tokenizer(batch_input, return_tensors="pt", padding=True).to('cuda')
            input_ids = input.input_ids
            attention_mask = input.attention_mask
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            item_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
        
        item_embeddings = torch.cat(item_embeddings, dim=0).cuda()
        dist = torch.cdist(item_embeddings, self.movie_embedding, p=2)
        
        indices = torch.argmin(dist, dim=1)
        
        movie_id = self.movie_index[indices].detach().cpu().numpy()
        
        movie_item = [self.id2item[str(int(id))] for id in movie_id]
        return movie_item
    
    def get_topk_near_item(self, item_list, k):
        item_embeddings = []
        task_list = [f'The type of {item} movie is' for item in item_list]
        for i, batch_input in tqdm(enumerate(batch(task_list, self.batch_size))):
            input = self.tokenizer(batch_input, return_tensors="pt", padding=True).to('cuda')
            input_ids = input.input_ids
            attention_mask = input.attention_mask
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            item_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
        
        item_embeddings = torch.cat(item_embeddings, dim=0).cuda()
        dist = torch.cdist(item_embeddings, self.movie_embedding, p=2)
        
        _, indices = torch.topk(dist, k, dim=1, largest=False)
        
        movie_id = self.movie_index[indices].detach().cpu().numpy()
        
        movie_item = [[self.id2item[str(int(id))] for id in id_list] for id_list in movie_id]
        return movie_item