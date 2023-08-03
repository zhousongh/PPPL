from dgllife.data import Tox21, ClinTox, PCBA, MUV, SIDER, ToxCast
from dgllife.utils import SMILESToBigraph
# from process_utils import ATOM_FEATURIZER, BOND_FEATURIZER
# from dataset.process_utils import ATOM_FEATURIZER, BOND_FEATURIZER
import torch
# from process_utils import getMolInitFeature, getSubInitFeature, getSubstructures
# from dataset.process_utils import getMolInitFeature, getSubInitFeature, getSubstructures
import os.path as osp
import pickle
import dgl
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
import pandas as pd
# from process_utils2 import ATOM_FEATURIZER, BOND_FEATURIZER, smiles_to_hypergraph
from dataset.process_utils2 import ATOM_FEATURIZER, BOND_FEATURIZER, smiles_to_hypergraph


DATASET_LIST = ['tox21', 'pcba', 'muv', 'clintox', 'sider', 'toxcast']


def save_obj(path, obj):
    # 序列化对象
    serialized = pickle.dumps(obj)
    with open(path, "wb") as file:
        file.write(serialized)


def load_obj(path):
    # 反序列化对象
    with open(path, "rb") as file:
        serialized = file.read()
    return pickle.loads(serialized)


class FlexDataset(Dataset):
    def __init__(self, dataset_name=None, root=None, device=None):

        super().__init__()

        # 检查数据集的名称是否正确
        assert dataset_name in DATASET_LIST
        # 检查根路径是否给出
        assert dataset_name != None and root != None and device != None

        self.dataset_name = dataset_name
        self.org_root = root
        self.device = device

        # 判断数据是否已经预处理
        if self.has_processed():
            print(f"{self.dataset_name} has been processed!")
        else:
            raw_dataset = self.load_raw_dataset()  # 加载原始数据集
            self.process(raw_dataset)  # 特征加工
        self.G_set = self.load_Gset()
        self.smiles, self.labels = self.load_smiles_and_labels()
        self.atom_to_sub = self.load_ats_set()
        self.masks = self.load_mask_set()

    def has_downloaded(self):
        file_path = osp.join(
            self.org_root, fr"raw/{self.dataset_name}_dglgraph.bin")
        return osp.exists(file_path)

    def has_processed(self):
        file_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_Graph.bin")
        return osp.exists(file_path)

    def __len__(self):
        return len(self.G_set)

    def __getitem__(self, idx):
        res = {}
        # 获取指定索引的数据，返回全局分子图和子结构图以及对应的标签0
        res['graph'] = self.G_set[idx]
        res['label'] = self.labels[idx]
        res['ats'] = self.atom_to_sub[idx]
        return res

    def load_Gset(self):
        assert self.has_processed()
        # 加载全图数据
        print('loading graphs...')
        file_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_Graph.bin")
        G_dataset = load_obj(file_path)
        return G_dataset

    def load_smiles_and_labels(self):
        assert self.has_processed()
        # 加载smiles和label
        print('loading smiles and labels...')
        smiles_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_smiles.bin")
        labels_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_labels.bin")
        smiles_list, labels_list = load_obj(smiles_path), load_obj(labels_path)
        return smiles_list, labels_list

    def load_ats_set(self):
        assert self.has_processed()
        # 加载全图数据
        print('loading the mapping matrix from atom to substructure...')
        file_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_ats.bin")
        ats = load_obj(file_path)
        return ats

    def load_mask_set(self):
        assert self.has_processed()
        # 加载全图数据
        print('loading masks...')
        file_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_mask.bin")
        mask = load_obj(file_path)
        return mask

    def load_raw_dataset(self):
        dataset = None
        save_path = osp.join(
            self.org_root, fr"raw/{self.dataset_name}_dglgraph.bin")
        # 根据数据集名称自适应加载不同数据集
        if self.dataset_name == 'tox21':
            # 检查数据集是否存在选择直接读取本地资源或者下载网络资源
            if self.has_downloaded():
                dataset = Tox21(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                edge_featurizer=BOND_FEATURIZER), load=True, cache_file_path=save_path)
            else:
                dataset = Tox21(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                edge_featurizer=BOND_FEATURIZER), load=False, cache_file_path=save_path)

        elif self.dataset_name == 'pcba':
            if self.has_downloaded():
                dataset = PCBA(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                               edge_featurizer=BOND_FEATURIZER), load=True, cache_file_path=save_path)
            else:
                dataset = PCBA(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                               edge_featurizer=BOND_FEATURIZER), load=False, cache_file_path=save_path)

        elif self.dataset_name == 'muv':
            if self.has_downloaded():
                dataset = MUV(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                              edge_featurizer=BOND_FEATURIZER), load=True, cache_file_path=save_path)
            else:
                dataset = MUV(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                              edge_featurizer=BOND_FEATURIZER), load=False, cache_file_path=save_path)

        elif self.dataset_name == 'clintox':
            if self.has_downloaded():
                dataset = ClinTox(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                  edge_featurizer=BOND_FEATURIZER), load=True, cache_file_path=save_path)
            else:
                dataset = ClinTox(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                  edge_featurizer=BOND_FEATURIZER), load=False, cache_file_path=save_path)

        elif self.dataset_name == 'toxcast':
            if self.has_downloaded():
                dataset = ToxCast(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                                  edge_featurizer=BOND_FEATURIZER), load=True, cache_file_path=save_path)
            else:
                dataset = ToxCast(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                                  edge_featurizer=BOND_FEATURIZER), load=False, cache_file_path=save_path)
        elif self.dataset_name == 'sider':
            if self.has_downloaded():
                dataset = SIDER(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                edge_featurizer=BOND_FEATURIZER), load=True, cache_file_path=save_path)
            else:
                dataset = SIDER(SMILESToBigraph(node_featurizer=ATOM_FEATURIZER,
                                edge_featurizer=BOND_FEATURIZER), load=False, cache_file_path=save_path)
        return dataset

    # 加工数据集并保存
    def process(self, raw_dataset):
        print(f'start processing dataset {self.dataset_name}...')

        # 处理分子图特征
        Graphs, smiles_list, label_list, ats_list, mask_list = [], [], [], [], []
        for _, data in tqdm(enumerate(iter(raw_dataset))):
            smiles, g, label, mask = data[0], None, data[2].to(
                self.device), data[3].to(self.device)
            try:
                g, atom_to_sub = smiles_to_hypergraph(smiles=smiles)
                Graphs.append(g)
                smiles_list.append(smiles)
                label_list.append(label)
                ats_list.append(atom_to_sub)
                mask_list.append(mask)
            except Exception as e:
                print(e)

        # 设置保存路径
        Graphs_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_Graph.bin")
        smiles_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_smiles.bin")
        labels_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_labels.bin")
        ats_path = osp.join(
            self.org_root, fr"processed/{self.dataset_name}_ats.bin")
        mask_path=osp.join(
            self.org_root, fr"processed/{self.dataset_name}_mask.bin")

        # 保存
        print('saving the results...')
        save_obj(Graphs_path, Graphs)
        save_obj(smiles_path, smiles_list)
        save_obj(labels_path, label_list)
        save_obj(ats_path, ats_list)
        save_obj(mask_path, mask_list)

    def task_pos_weights(self):
        assert self.dataset_name in DATASET_LIST
        labels,masks = torch.stack(self.labels),torch.stack(self.masks)
        task_pos_weight = torch.ones(labels.shape[1]).to(self.device)
        num_pos = torch.sum(labels, dim=0)
        num_ex=torch.sum(masks,dim=0)
        task_pos_weight[num_pos > 0] = (
            (num_ex - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weight


class FlexDataLoader(DataLoader):
    def __init__(self, device, *args, **kwargs):
        self.device = device
        kwargs['collate_fn'] = self.collate
        super(FlexDataLoader, self).__init__(*args, **kwargs)

    def collate(self, batch):
        '''重新定义批处理行为'''
        G_list, label_list = [], []
        for item in batch:
            G_list.append(item['graph'])
            label_list.append(item['label'])
        label_batch = torch.stack(
            label_list).reshape(-1) if label_list[0].numel() == 1 else torch.stack(label_list)
        return (dgl.batch(G_list).to(self.device), label_batch.to(self.device))


if __name__ == "__main__":
    # 测试
    dataset = FlexDataset(dataset_name='clintox',
                          root=r'/mnt/klj/PPPL/dataset/data', device=torch.device("cuda"))
    print(f'GLOBAL:{dataset[0]["graph"].nodes["func_group"]}')
    print(f'LABEL:{dataset[0]["label"]}')
    print(f'DATASET SIZE:{len(dataset)}')
    print(f'G_SET SIZE:{len(dataset.G_set)}')
    print(f'LABEL SIZE:{len(dataset.labels)}')
    print(f'SMILES SIZE:{len(dataset.smiles)}')

    loader = FlexDataLoader(dataset=dataset,device=torch.device("cuda"), batch_size=128, shuffle=True)
    for i in range(10):
        print(next(iter(loader)))
        break
    # print(len(dataset.getAllSubstructure()))