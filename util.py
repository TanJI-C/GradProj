# 实现根据查询的workload和query statement获得plan tree
import pandas as pd
import json
import psycopg2
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def q_error(pred, target):
    return torch.max(pred / target, target / pred)

def get_statistics_of_workloads(plans, target_path):
    runtimes = []
    cards = []
    costs = []
    node_types = set()

    for plan in plans:
        plan = plan['Plan']
        runtimes.append(plan['Actual Total Time'])
        costs.append(plan['Total Cost'])
        cards.append(plan['Plan Rows'])
        node_types.add(plan['Node Type'])

        stack = [plan]
        while len(stack) > 0:
            node = stack.pop()
            if 'Plans' in node:
                for child in node['Plans']:
                    runtimes.append(child['Actual Total Time'])
                    costs.append(child['Total Cost'])
                    cards.append(child['Plan Rows'])
                    node_types.add(child['Node Type'])
                    stack.append(child)

    runtimes = np.array(runtimes)
    costs = np.array(costs)
    cards = np.array(cards)
    node_types = list(node_types)

    statistics = {
        "Actual Total Time": {
            "type": 'numeric',
            "max": float(np.max(runtimes)),
            "min": float(np.min(runtimes)),
            "center": float(np.median(runtimes)),
            "scale": float(np.quantile(runtimes, 0.75)) - float(np.quantile(runtimes, 0.25)),
        },
        "Plan Rows": {
            "type": 'numeric',
            "max": float(np.max(cards)),
            "min": float(np.min(cards)),
            "center": float(np.median(cards)),
            "scale": float(np.quantile(cards, 0.75)) - float(np.quantile(cards, 0.25)),
        },
        "Total Cost": {
            "type": 'numeric',
            "max": float(np.max(costs)),
            "min": float(np.min(costs)),
            "center": float(np.median(costs)),
            "scale": float(np.quantile(costs, 0.75)) - float(np.quantile(costs, 0.25)),
        },
        "node_types": {
            "type": 'categorical',
            "value_dict": {node_type: i for i, node_type in enumerate(node_types)},
        },
    }

    with open(target_path, 'w') as f:
        json.dump(statistics, f)

class TreeNode:
    def __init__(self, time_stamp, nodeType, cost, card, join, filt, label, dep) -> None:
        self.index = time_stamp
        self.nodeType = nodeType
        self.cost = cost
        self.card = card
        self.join = join
        self.filt = filt
        self.label = label
        self.dep = dep

        self.children = []
        self.parent = None

# Encoding the features of the plan
class Encoder:

    # 使用one-hot对node_type进行编码
    # 使用scaler对估计代价和行数进行预处理
    # 同时得到attention_mask: 去掉pdding数据的影响以及保留树型特征（只保留有父子关系）
    def __init__(self, feature_statistics, configs):
        # onehot
        self.type_to_one_hot_dict = {}
        type_to_index_dict = feature_statistics['node_types']['value_dict']
        type_num = len(type_to_index_dict)
        print(type_to_index_dict)
        for type_name, index_value in type_to_index_dict.items():
            self.type_to_one_hot_dict[type_name] = np.zeros((1, type_num), dtype=np.int32)
            self.type_to_one_hot_dict[type_name][0][index_value] = 1
        
        # scalers
        self.scaler_dict = {}
        for k, v in feature_statistics.items():
            if v['type'] == "numeric":
                scaler = RobustScaler()
                scaler.center_ = v['center']
                scaler.scale_ = v['scale']
                self.scaler_dict[k] = scaler
        # test
        print(self.scaler_dict)

        # MAX_time
        self.max_cost = feature_statistics['Actual Total Time']['max']

        self.pad_length = configs.pad_length
        self.node_length = configs.node_length
        
        self.padding_value_for_feature = 0
        self.padding_value_for_label = 1
        self.loss_weight = 0.5
    # 从trees_nodes中提取特征: 包括type+row+cost、atten_mask以及label(run time)
    def node2feature(self, trees_nodes):
        input_feature = []
        label = []
        atten_mask = []
        loss_mask = []
        for tree_nodes_item in trees_nodes: # enumerate all plan trees
            # get feature
            input_feature_tmp = []
            label_tmp = []
            seq_length = len(tree_nodes_item)
            loss_mask_tmp = np.zeros((self.pad_length))
            # get loss mask
            for index, tree_node_item in enumerate(tree_nodes_item): # enumerate all nodes of current trees
                input_feature_tmp.append(self.type_to_one_hot_dict[tree_node_item.nodeType])
                input_feature_tmp.append(self.scaler_dict['Plan Rows'].transform(
                    np.array([[tree_node_item.card]])
                ))
                input_feature_tmp.append(self.scaler_dict['Total Cost'].transform(
                    np.array([[tree_node_item.cost]])
                ))
                loss_mask_tmp[index] = np.power(self.loss_weight, tree_node_item.dep)
                label_tmp.append(tree_node_item.label)

            ## pad feature
            # became (1 X seq_length*node_length)
            input_feature_tmp = torch.from_numpy(np.concatenate(input_feature_tmp, axis = 1))
            input_feature_tmp = torch.nn.functional.pad(
                input_feature_tmp,
                (0, self.pad_length * self.node_length - input_feature_tmp.shape[1]),
                value = self.padding_value_for_feature
            )
            input_feature.append(input_feature_tmp)
            
            # pad_length label
            label_tmp = torch.tensor(label_tmp)
            label_tmp = torch.nn.functional.pad(
                label_tmp,
                (0, self.pad_length - label_tmp.shape[0]),
                value = self.padding_value_for_label
            )
            label.append(label_tmp / self.max_cost + 1e-7)
            '''
            # one label
            label.append(torch.tensor([tree_nodes_item[0].label / self.max_cost + 1e-7]))
            ''' 

            # pad_length loss_mask
            loss_mask_tmp = torch.from_numpy(loss_mask_tmp)
            loss_mask.append(loss_mask_tmp)
            '''
            # one loss_mask
            loss_mask.append(torch.tensor([np.power(self.loss_weight, tree_nodes_item[0].dep)]))
            '''

            # get atten_mask
            atten_tuples = self.dfs2atten_recursive(tree_nodes_item[0], [])
            atten_mask_tmp = torch.ones((self.pad_length, self.pad_length))
            for u, v in atten_tuples:
                atten_mask_tmp[u][v] = 0
            # attention itself
            for idx in range(self.pad_length):
                atten_mask_tmp[idx][idx] = 0
            # test 
            atten_mask.append(atten_mask_tmp)


        input_feature = torch.stack(input_feature).to(dtype=torch.float32)
        atten_mask = torch.stack(atten_mask).to(dtype=torch.bool)
        label = torch.stack(label).to(dtype=torch.float32)
        loss_mask = torch.stack(loss_mask).to(dtype=torch.float32)
        return input_feature, atten_mask, label, loss_mask



    def dfs2atten_recursive(self, root: TreeNode, grandfa) -> List[Tuple]:
        index = root.index
        result = []
        for u in grandfa:
            result.append((u, index))
        grandfa.append(index)
        
        for u in root.children:
            result_son = self.dfs2atten_recursive(u, grandfa)
            result.extend(result_son)
        grandfa.pop()

        return result
    
    # 将join的条件提取出来，并
    def format_join(plan):
        join = None
        if 'Hash Join' in plan:
            join = plan['Hash Cond']
        elif 'Join Filter' in plan:
            join = plan['Join Filter']
        # TODO: inedx cond
        # elif 'Index Cond' in plan and not plan['Index Cond'][-2].isnumeric():
        #     join = plan['Index Cond']
        else:
            return None
        
        pl = plan
        while 'Alias' not in pl:
            if 'parent' not in pl:
                break
            pl = pl['parent']
        alias = None
        if 'Alias' in pl:
            alias = pl['Alias']
        cols = join[1:-1].split(' = ')
        cols = [plan['Alias'] + '.' + col if len(col.split('.')) == 1
               else col for col in cols]
        join = sorted(cols)
        return join

    def format_filter(plan):
        filter = []
        if 'Filter' in plan:
            filter.append(plan['Filter'])
        if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
            filter.append(plan['Index Cond'])
        if 'Recheck Cond' in plan:
            filter.append(plan['Recheck Cond'])
        
        if len(filter) == 0:
            return None
        return filter

    # format data and return plan trees json
    def format_imdb(files_path) -> List[dict]:
        if isinstance(files_path, str):
            files_path = [files_path]
        nodes = []
        for db_id, file_name in enumerate(files_path):
            df = pd.read_csv(file_name)
            for plan in df['json']: # 为每一行加上一个新的属性，方便分割数据集
                json_item = json.loads(plan)
                json_item['database_id'] = db_id
                nodes.append(json_item)
        return nodes
    # format data and return plan trees json
    def format_workload(files_path) -> List[dict]:
        if isinstance(files_path, str):
            files_path = [files_path]
        nodes = []
        for db_id, file_name in enumerate(files_path):
            with open(file_name) as jf:
                json_file = json.load(jf)
                for json_item in json_file:
                    json_item['database_id'] = db_id
                    nodes.append(json_item)
        return nodes

    def format_imdb_test(files_path):
        if isinstance(files_path, str):
            files_path = [files_path]
        nodes = []
        for db_id, file_name in enumerate(files_path):
            with open(file_name) as jf:
                json_file = json.load(jf)
                for json_item in json_file:
                    json_item_tmp = json_item["plan"][0][0][0]
                    if json_item_tmp["Plan"]["Actual Total Time"] < 100:
                        continue
                    json_item_tmp['database_id'] = db_id
                    nodes.append(json_item_tmp)
        return nodes
        


# dataset
class PlanTreeDataSet(Dataset):
    def __init__(self, json_data: List[dict], encoder: Encoder) -> None:
        self._dataset_size = len(json_data)
        self._json_data = json_data
        self._trees_nodes = []
        self._database_id_of_plan = []
        self.dfs_plan_trees()
        self._feature, self._atten_mask, self._cost_label, self._loss_mask = encoder.node2feature(self._trees_nodes)
        print(self._feature.size(), self._atten_mask.size(), self._cost_label.size(), self._loss_mask.size())
        # 类型说明：final_feature: torch.tensor(torch.tensor(torch.tensor(seq_length * feature))) (Plans, 1, pad_length * node_length)
        # 类型说明：cost_label: torch.tensor(torch.tensor(cost))     (Plans, act_cost)
    def __getitem__(self, index) -> Any:
        return self._feature[index], self._atten_mask[index], self._loss_mask[index], self._cost_label[index], self._database_id_of_plan[index]

    def __len__(self):
        return self._dataset_size

    def dfs_plan_trees(self) -> None:
        for item in self._json_data:
            self._tree_nodes_tmp = []
            self._time_stamp = 0
            self.dfs_plan_tree_recursive(item['Plan'], 0)
            self._trees_nodes.append(self._tree_nodes_tmp)
            self._database_id_of_plan.append(item['database_id'])
    # 遍历plan tree返回根节点TreeNode
    def dfs_plan_tree_recursive(self, plan, dep) -> TreeNode:
        nodeType = plan['Node Type']
        cost = plan['Total Cost']
        card = plan['Plan Rows']
        label = plan['Actual Total Time']
        join = Encoder.format_join(plan)
        filt = None #Encoder.forma_filter(plan)
        
        root = TreeNode(self._time_stamp, nodeType, cost, card, join, filt, label, dep)
        self._time_stamp = self._time_stamp + 1

        self._tree_nodes_tmp.append(root)

        if 'Plans' in plan:
            for subplan in plan['Plans']:
                child_node = self.dfs_plan_tree_recursive(subplan, dep + 1)
                child_node.parent = root
                root.children.append(child_node)
        
        return root
    

def get_dataloader(data_set, configs):
    train_data, test_data = [], []
    test_database_ids = configs.test_database_ids
    for plan_meta in data_set:
        if plan_meta[-1] in test_database_ids:
            test_data.append(plan_meta[:-1])
        else:
            train_data.append(plan_meta[:-1])
    # test_data = train_data
    print(len(data_set[0]), len(train_data[0]), len(test_data[0]))
    # split train_data into train_data and val_data by 9:1
    train_data, val_data = train_test_split(
        train_data, test_size=0.1, random_state=configs.random_seed
    )

    bs = configs.batch_size
    train_dataloader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=bs, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

