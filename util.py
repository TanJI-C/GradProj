# 实现根据查询的workload和query statement获得plan tree
import json
import psycopg2
import torch
from torch.utils.data import Dataset
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

class DBManager:
    def __init__(self, db_configuration) -> None:
        self.db_configuration = db_configuration
        self._conn = psycopg2.connect(
            dbname=db_configuration.dbname,
            user=db_configuration.user,
            password=db_configuration.password,
            host=db_configuration.host
        )
        self._cursor = self._conn.cursor()
        self._cursor.execute(f"select setseed({db_configuration.seed})")
        self._cursor.execute("load  'pg_hint_plan")
    def _configuration_switch(self, configuration: Optional[Sequence[str]] = None) -> None:
        if configuration is None:
            return
        for parameter in configuration:
            self._cursor.execute(f"show {parameter}")
            val = self._cursor.fetchone()[0]
            new_val = 'OFF' if val == 'ON' else 'ON'
            self._cursor.execute(f"set {parameter} TO {new_val}")

    def get_query_plan_tree(self, query: str, configuration: Optional[Sequence[str]] = None): #explain语句会同时执行查询吗？
        self._configuration_switch(configuration)
        explain_query = f"explain (format json, analyze, buffers) {query}"
        self._cursor.execute(explain_query)
        return self._cursor.fetchone()[0][0]

class TreeNode:
    def __init__(self, nodeType, cost, card, join, filt) -> None:
        self.nodeType = nodeType
        self.cost = cost
        self.card = card
        self.join = join
        self.filt = filt

        self.children = []
        self.parent = None

class Encoder:
    padding_bit             =   0B0
    seq_scan_bit            =   0B1
    index_scan_bit          =   0B10
    bitmap_index_scan_bit   =   0B100
    bitmap_heap_scan_bit    =   0B1000
    nested_loop_bit         =   0B10000
    hash_join_bit           =   0B100000
    merge_join_bit          =   0B1000000
    hash_bit                =   0B10000000
    sort_bit                =   0B100000000
    limit_bit               =   0B1000000000
    aggregate_bit           =   0B10000000000
    gather_bit              =   0B100000000000
    gather_merge_bit        =   0B1000000000000
    materialize_bit         =   0B10000000000000
    other_type_bit          =   0B100000000000000

    node_length = 40        #对齐长度

    type_dict = {
        'Padding':              padding_bit,
        'Seq Scan':             seq_scan_bit,
        'Index Scan':           index_scan_bit,
        'Bitmap Index Scan':    bitmap_index_scan_bit,
        'Bitmap Heap Scan':     bitmap_heap_scan_bit,
        'Nested Loop':          nested_loop_bit,
        'Hash Join':            hash_join_bit,
        'Merge Join':           merge_join_bit,
        'Hash':                 hash_bit,
        'Sort':                 sort_bit,
        'Limit':                limit_bit,
        'Aggregate':            aggregate_bit,
        'Gather':               gather_bit,
        'Gather Merge':         gather_merge_bit,
        'Materialize':          materialize_bit,
        'Other Type':           other_type_bit
    }
    typebit2tensor_dict = {
        '0b0':                  torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b1':                  torch.tensor([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b10':                 torch.tensor([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b100':                torch.tensor([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b1000':               torch.tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b10000':              torch.tensor([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b100000':             torch.tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b1000000':            torch.tensor([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b10000000':           torch.tensor([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], dtype=torch.float32),
        '0b100000000':          torch.tensor([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], dtype=torch.float32),
        '0b1000000000':         torch.tensor([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], dtype=torch.float32),
        '0b10000000000':        torch.tensor([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], dtype=torch.float32),
        '0b100000000000':       torch.tensor([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], dtype=torch.float32),
        '0b1000000000000':      torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], dtype=torch.float32),
        '0b10000000000000':     torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], dtype=torch.float32),
        '0b100000000000000':    torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=torch.float32),
    }

    # 将join的条件提取出来，并
    def format_join(plan):
        join = None
        if 'Hash Join' in plan:
            join = plan['Hash Cond']
        elif 'Join Filter' in plan:
            join = plan['Join Filter']
        elif 'Index Cond' in plan and not plan['Index Cond'][-2].isnumeric():
            join = plan['Index Cond']
        else:
            return None
        
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

    def type_encode(node_sequence: List[TreeNode]):
        feature_sequence = []
        for item in node_sequence:
            if item.nodeType in Encoder.type_dict:
                feature_sequence.append(Encoder.typebit2tensor_dict[bin(Encoder.type_dict[item.nodeType])])
            else:
                feature_sequence.append(Encoder.typebit2tensor_dict[bin(Encoder.type_dict['Other Type'])])
        # print(feature_sequence)
        if len(node_sequence) > Encoder.node_length:
            print("长度超限,{}".format(len(node_sequence)))
        for _ in range(Encoder.node_length - len(node_sequence)):
            feature_sequence.append(Encoder.typebit2tensor_dict[bin(Encoder.type_dict['Padding'])])
        

        feature_sequence = torch.stack(feature_sequence)
        return feature_sequence
    
    def format_imdb(json_df: pd.DataFrame) -> List[dict]:
        length = len(json_df)
        nodes = [json.loads(plan) for plan in json_df['json']]
        return nodes



        
class BatchData(dict):
    # def __init__(self, type_feature, info_feature) -> None:
    #     super(BatchData, self).__init__()
    #     self.type_features = type_feature
    #     self.info_features = info_feature
    
    def to(self, device):
        # self.type_feature = self.type_feature
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self

class PlanTreeDataSet(Dataset):
    def __init__(self, json_data: List[dict]) -> None:
        self._dataset_size = len(json_data)
        self._json_data = json_data
        self._tree_nodes_tmp = []
        self._trees_nodes = []
        self._type_feature = []
        self._info_feature = []
        self._cost_label = []
        self.dfs_plan_trees()
        self.node2feature()
        # 类型说明：type_feature: torch.tensor(torch.tensor(torch.tensor(one-hot for node type))) (Plans, nodes, one-hot)
        # 类型说明：info_feature: torch.tensor(torch.tensor([cost, row]))   (Plans, [cost, row])
        # 类型说明：cost_label: torch.tensor(torch.tensor([cost, row]))     (Plans, act_cost)

    def __getitem__(self, index) -> Any:
        return BatchData({
            "type_feature": self._type_feature[index], 
            "info_feature": self._info_feature[index]
            }), torch.tensor([self._cost_label[index]], dtype=torch.float32)

    def __len__(self):
        return self._dataset_size

    def dfs_plan_trees(self) -> None:
        for item in self._json_data:
            self._tree_nodes_tmp = []
            self.dfs_plan_tree_recursive(item['Plan'])
            self._trees_nodes.append(self._tree_nodes_tmp)
    # 遍历plan tree返回根节点TreeNode
    def dfs_plan_tree_recursive(self, plan) -> TreeNode:
        nodeType = plan['Node Type']
        cost = None #[plan['Actual Startup Time'], plan['Actual Total Time']]
        card = None #plan['Actual Rows']
        join = Encoder.format_join(plan)
        filt = None #Encoder.forma_filter(plan)
        
        root = TreeNode(nodeType, cost, card, join, filt)

        self._tree_nodes_tmp.append(root)

        if 'Plans' in plan:
            for subplan in plan['Plans']:
                child_node = self.dfs_plan_tree_recursive(subplan)
                child_node.parent = root
                root.children.append(child_node)
        
        return root
    
    # get feature from tree_node sequence
    def node2feature(self):
        for tree_nodes_item, json_item in zip(self._trees_nodes, self._json_data):
            self._type_feature.append(Encoder.type_encode(tree_nodes_item))
            self._info_feature.append(torch.tensor([json_item['Plan']['Startup Cost'] + json_item['Plan']['Total Cost'],
                                json_item['Plan']['Plan Rows']], dtype=torch.float32))
            self._cost_label.append(json_item['Plan']['Actual Startup Time'] + json_item['Plan']['Actual Total Time'])
        # print(self._type_feature)
        # print(self._info_feature)
        self._type_feature = torch.stack(self._type_feature)
        self._info_feature = torch.stack(self._info_feature)
        