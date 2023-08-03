import torch
from dgllife.utils import mol_to_bigraph, smiles_to_bigraph
from dgllife.data import Tox21, BACE, BBBP, ClinTox, ESOL, FreeSolv, Lipophilicity, SIDER
from dgllife.utils import SMILESToBigraph
from rdkit import Chem
from functools import partial
from tqdm import tqdm
from rdkit.Chem import BRICS
from rdkit.Chem import Recap
from dgllife.utils import BaseAtomFeaturizer, BaseBondFeaturizer
from dgllife.utils.featurizers import (
    atom_chirality_type_one_hot,
    atom_explicit_valence_one_hot,
    atom_hybridization_one_hot,
    atom_is_aromatic_one_hot,
    atom_is_chiral_center,
    atom_is_in_ring_one_hot,
    atom_total_num_H_one_hot,
    atom_type_one_hot,
    atomic_number,
    bond_is_conjugated_one_hot,
    bond_is_in_ring,
    bond_type_one_hot,
    bond_stereo_one_hot
)


ATOM_FEATURIZER = BaseAtomFeaturizer({'atom_type': partial(atom_type_one_hot,
                                                           allowable_set=["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"]),
                                      'atomic_number': atomic_number,
                                      'atom_explicit': atom_explicit_valence_one_hot,
                                      'atom_num_H': atom_total_num_H_one_hot,
                                      'atom_hybridization': atom_hybridization_one_hot,
                                      'aromatic': atom_is_aromatic_one_hot,
                                      'atom_in_ring': atom_is_in_ring_one_hot,
                                      'atom_chirality': atom_chirality_type_one_hot,
                                      'atom_chiral_center': atom_is_chiral_center})

BOND_FEATURIZER = BaseBondFeaturizer({'bond_type': bond_type_one_hot,
                                      'in_ring': bond_is_in_ring,
                                      'conj_bond': bond_is_conjugated_one_hot,
                                      'bond_stereo': bond_stereo_one_hot})

NODE_ATTRS = ['atom_type',
              'atomic_number',
              'atom_explicit',
              'atom_num_H',
              'atom_hybridization',
              'aromatic',
              'atom_in_ring',
              'atom_chirality',
              'atom_chiral_center']
EDGE_ATTRS = ['bond_type',
              'in_ring',
              'conj_bond',
              'bond_stereo']


def generate_nf_and_ef(graph, device):
    # 将分子图的各个化学性质的one_hot编码特征拼接，获得初始的原子特征与化学键特征
    try:
        nf = torch.cat([graph.ndata[nf_field]
                       for nf_field in NODE_ATTRS], dim=-1)
        nf = nf.to(device)
        graph.ndata['feat'] = nf
    except KeyError as e:  # Ionic bond only case.
        raise e
    try:
        ef = torch.cat([graph.edata[ef_field]
                       for ef_field in EDGE_ATTRS], dim=-1)
        ef = ef.to(device)
        graph.edata['feat'] = ef
    except KeyError as e:  # Ionic bond only case.
        raise e
    return graph


def getSubstructures(mol=None, smile=None, decomp='brics'):
    # 获取指定分子的子结构
    assert mol is not None or smile is not None,\
        'need at least one info of mol'
    assert decomp in ['brics', 'recap'], 'Invalid decomposition method'
    if mol is None:
        mol = Chem.MolFromSmiles(smile)

    if decomp == 'brics':
        substructures = BRICS.BRICSDecompose(mol)
    else:
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys())
    return substructures


def substructureToGraph(substructures, device):
    # 将子结构转化为分子图
    subG_list = []
    for sub in substructures:
        g = smiles_to_bigraph(smiles=sub, node_featurizer=ATOM_FEATURIZER,
                              edge_featurizer=BOND_FEATURIZER)
        assert g != None
        g = g.to(device)
        subG_list.append(g)
    return subG_list


def getSubInitFeature(smiles_list, device):
    # 处理出整个数据集的分子图的子结构以及子结构的特征,返回处理后的子结构信息
    subG_set = []
    print("processing substructure features...")
    for _, smiles in tqdm(enumerate(iter(smiles_list))):
        substructures = getSubstructures(smile=smiles)
        subG_list = substructureToGraph(
            substructures=substructures, device=device)
        new_list = []
        for subg in subG_list:
            try:
                subg = generate_nf_and_ef(subg, device)
                new_list.append(subg)
            except Exception as e:
                print(e)
        subG_set.append(new_list)
    return subG_set


def getMolInitFeature(dataset, device):
    # 处理出整个数据集的分子图的初始特征
    print("processing global features...")
    graphs, smiles_list, label_list = [], [], []
    for _, data in tqdm(enumerate(iter(dataset))):
        smiles, g, label = data[0], data[1].to(device), data[2].to(device)
        try:
            g = generate_nf_and_ef(g, device)
            graphs.append(g)
            smiles_list.append(smiles)
            label_list.append(label)
        except Exception as e:
            print(e)
    return graphs, smiles_list, label_list
