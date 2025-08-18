import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from .descriptors.rdNormalizedDescriptors import RDKit2DNormalized
import numpy as np
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from rdkit.Chem.Scaffolds import MurckoScaffold

def one_hot_vector(val, lst, add_unknown=True):
    if add_unknown:
        vec = np.zeros(len(lst) + 1)
    else:
        vec = np.zeros(len(lst))

    vec[lst.index(val) if val in lst else -1] = 1
    return vec


def get_atom_features(atom, d_atom):
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])
    v2 = one_hot_vector(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])
    v3 = [
        atom.GetTotalNumHs(includeNeighbors=True) / 8,
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 8,
        atom.GetTotalValence() / 8,
        0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
        0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ]

    v4 = [
        atom.GetIdx() + 1
    ]

    attributes = np.concatenate([v1, v2, v3, v4], axis=0)

    assert len(attributes) == d_atom + 1

    return attributes

def get_bond_features(bond, d_edge):
    v1 = one_hot_vector(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], add_unknown=False)

    v2 = one_hot_vector(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOANY,
                                           Chem.rdchem.BondStereo.STEREOCIS,
                                           Chem.rdchem.BondStereo.STEREOE,
                                           Chem.rdchem.BondStereo.STEREONONE,
                                           Chem.rdchem.BondStereo.STEREOTRANS,
                                           Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)

    v3 = [
        int(bond.GetIsConjugated()),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]

    attributes = np.concatenate([v1, v2, v3])

    assert len(attributes) == d_edge
    return attributes

def get_molecular_descriptor_features(mol, d_atom):
    generator = RDKit2DNormalized()
    descriptors = generator.process(Chem.MolToSmiles(mol))
    descriptors = np.array(descriptors[1:], dtype=float)
    descriptors = np.nan_to_num(descriptors, nan=0.0)

    if len(descriptors) >= d_atom:
        global_information_node = descriptors[:d_atom]
    else:
        global_information_node = np.pad(descriptors, (0, d_atom - len(descriptors)), mode='constant', constant_values=0)
    return global_information_node

def load_data_from_mol(mol, d_atom, d_edge, max_length):
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    AllChem.ComputeGasteigerCharges(mol)

    node_features = np.array([get_atom_features(atom, d_atom) for atom in mol.GetAtoms()])

    global_information_node = get_molecular_descriptor_features(mol, d_atom)
    global_information_node = global_information_node.reshape(1, -1)

    num_atoms = mol.GetNumAtoms()
    position_index = num_atoms + 1
    global_information_node = np.concatenate([global_information_node, [[position_index]]], axis=1)

    node_features = np.vstack([node_features, global_information_node])

    bond_features = np.zeros((num_atoms, num_atoms, d_edge))
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()
        bond_feat = get_bond_features(bond, d_edge)
        bond_features[begin_atom_idx, end_atom_idx, :] = bond_feat
        bond_features[end_atom_idx, begin_atom_idx, :] = bond_feat

    bond_features = np.pad(bond_features, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
    global_edge_features = np.zeros((d_edge,))
    for i in range(num_atoms):
        bond_features[i, -1, :] = global_edge_features
        bond_features[-1, i, :] = global_edge_features

    adjacency_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(float)
    adjacency_matrix = np.pad(adjacency_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    adjacency_matrix[-1, :-1] = 1
    adjacency_matrix[:-1, -1] = 1

    padded_node_features = pad_array(node_features, (max_length + 1, node_features.shape[-1]))
    padded_bond_features = pad_array(bond_features, (max_length + 1, max_length + 1, bond_features.shape[-1]))
    padded_adjacency_matrix = pad_array(adjacency_matrix, (max_length + 1, max_length + 1))

    return padded_node_features, padded_bond_features, padded_adjacency_matrix

class Molecule:
    def __init__(self, mol, label, d_atom, d_edge, max_length):
        self.smile = Chem.MolToSmiles(mol)
        self.label = label
        self.node_features, self.bond_features, self.adjacency_matrix = load_data_from_mol(mol, d_atom, d_edge, max_length)        

class MolDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataSet(self.data_list[key])
        return self.data_list[key]


def mask_node_features(node_features, mask_ratio, include_global_node=False):
    node_features = node_features.copy()
    num_nodes = np.count_nonzero(np.sum(np.abs(node_features), axis=-1))
    if not include_global_node:
        num_nodes -= 1

    if num_nodes <= 0:
        masked_labels = np.zeros_like(node_features)
        return node_features, masked_labels
    mask_num = max(1, int(num_nodes * mask_ratio))
    if include_global_node:
        mask_indices = np.random.choice(num_nodes, mask_num, replace=False)
    else:
        mask_indices = np.random.choice(num_nodes, mask_num, replace=False)
    masked_labels = np.zeros_like(node_features)
    masked_labels[mask_indices] = node_features[mask_indices]
    node_features[mask_indices] = 0
    return node_features, masked_labels


def get_scaffold_mol(mol, node_features, bond_features, adjacency_matrix):
    num_nodes = node_features.shape[0]
    num_atoms = mol.GetNumAtoms()
    global_node_idx = num_atoms

    atom_mask = np.zeros((num_nodes,), dtype=int)

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold_mol)
    scaffold_atom_indices = set(match)

    for idx in scaffold_atom_indices:
        atom_mask[idx] = 1

    atom_mask[global_node_idx] = 1

    scaffold_node_features = node_features * atom_mask[:, None]
    mask_matrix = np.outer(atom_mask, atom_mask)
    scaffold_adjacency_matrix = adjacency_matrix * mask_matrix
    scaffold_bond_features = bond_features * mask_matrix[:, :, None]
    return scaffold_node_features, scaffold_bond_features, scaffold_adjacency_matrix

def get_functional_group_mol(mol, node_features, bond_features, adjacency_matrix):
    functional_groups_smarts = {
        "acetic anydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "acetylenic carbon": "[$([CX2]#C)]",
        "acyl bromide": "[CX3](=[OX1])[Br]",
        "acyl chloride": "[CX3](=[OX1])[Cl]",
        "acyl fluoride": "[CX3](=[OX1])[F]",
        "acyl iodide": "[CX3](=[OX1])[I]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "alkane": "[CX4]",
        "unbranched alkene": "[R0;D2,D1][R0;D2][R0;D2,D1]",
        "allenic carbon": "[$([CX2](=C)=C)]",
        "amide": "[NX3][CX3](=[OX1])[#6]",
        "amidium": "[NX3][CX3]=[NX3+]",
        "amino acid": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "azo nitrogen": "[NX2]=N",
        "azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "azoxy nitrogen": "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "diazene": "[NX2]=[NX2]",
        "diazo nitrogen": "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "bromine": "[Br]",
        "carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "carbamic ester": "[NX3][CX3](=[OX1])[OX2H0]",
        "carbamic acid": "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "carbo azosulfone": "[SX4](C)(C)(=O)=N",
        "carbo thiocarboxylate": "[S-][CX3](=S)[#6]",
        "carbo thioester": "S([#6])[CX3](=O)[#6]",
        "carboxylate ion": "[CX3](=O)[O-]",
        "carbonic acid": "[CX3](=[OX1])(O)O",
        "carbonic ester": "C[OX2][CX3](=[OX1])[OX2]C",
        "carbonyl group": "[CX3]=[OX1]",
        "carbonyl with carbon": "[CX3](=[OX1])C",
        "carbonyl with nitrogen": "[OX1]=CN",
        "carbonyl with oxygen": "[CX3](=[OX1])O",
        "carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
        "chlorine": "[Cl]",
        "cyanamide": "[NX3][CX2]#[NX1]",
        "di sulfide": "[#16X2H0][#16X2H0]",
        "enamine": "[NX3][CX3]=[CX3]",
        "enol": "[OX2H][#6X3]=[#6]",
        "ester": "[#6][CX3](=O)[OX2H0][#6]",
        "ether": "[OD2](C)C",
        "fluorine": "[F]",
        "hydrogen": "[H]",
        "hydrazine": "[NX3][NX3]",
        "hydrazone": "[NX3][NX2]=[*]",
        "hydroxyl in carboxylic acid": "[OX2H][CX3]=[OX1]",
        "isonitrile": "[CX1-]#[NX2+]",
        "imide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "imine": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "iminium": "[NX3+]=[CX3]",
        "ketone": "[#6][CX3](=O)[#6]",
        "peroxide": "[OX2,OX1-][OX2,OX1-]",
        "phenol": "[OX2H][cX3]:[c]",
        "phosphoric acid": "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "phosphoric ester": "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        "primary alcohol": "[OX2H]",
        "primary amine": "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "proton": "[H+]",
        "mono sulfide": "[#16X2H0][!#16]",
        "nitrate": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "nitrile": "[NX1]#[CX2]",
        "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "nitroso": "[NX2]=[OX1]",
        "n-oxide": "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        "secondary amine": "[NX3;H1;!$(NC=O)]",
        "sulfate": "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "sulfamate": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "sulfamic acid": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "sulfenic acid": "[#16X2][OX2H,OX1H0-]",
        "sulfenate": "[#16X2][OX2H0]",
        "sulfide": "[#16X2H0]",
        "sulfonate": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "sulfinate": "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "sulfinic acid": "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "sulfonamide": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "sulfone": "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "sulfonic acid": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "sulfoxide": "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "sulfur": "[#16!H0]",
        "sulfuric acid ester": "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "sulfuric acid diester": "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "thioamide": "[NX3][CX3]=[SX1]",
        "thiol": "[#16X2H]",
        "vinylic carbon": "[$([CX3]=[CX3])]",
    }

    functional_group_atom_indices = set()
    for fg_name, fg_smarts in functional_groups_smarts.items():
        fg_mol = Chem.MolFromSmarts(fg_smarts)
        if fg_mol is None:
            continue
        matches = mol.GetSubstructMatches(fg_mol)
        for match in matches:
            functional_group_atom_indices.update(match)

    num_nodes = node_features.shape[0]
    num_atoms = mol.GetNumAtoms()
    global_node_idx = num_atoms

    atom_mask = np.zeros((num_nodes,), dtype=int)

    for idx in functional_group_atom_indices:
        atom_mask[idx] = 1

    atom_mask[global_node_idx] = 1

    fg_node_features = node_features * atom_mask[:, None]
    mask_matrix = np.outer(atom_mask, atom_mask)
    fg_adjacency_matrix = adjacency_matrix * mask_matrix
    fg_bond_features = bond_features * mask_matrix[:, :, None]

    return fg_node_features, fg_bond_features, fg_adjacency_matrix

def get_pharmacophore_mol(mol, node_features, bond_features, adjacency_matrix):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    features = factory.GetFeaturesForMol(mol)
    pharmacophore_atom_indices = set()
    for feat in features:
        atom_ids = feat.GetAtomIds()
        pharmacophore_atom_indices.update(atom_ids)

    num_nodes = node_features.shape[0]
    num_atoms = mol.GetNumAtoms()
    global_node_idx = num_atoms

    atom_mask = np.zeros((num_nodes,), dtype=int)

    for idx in pharmacophore_atom_indices:
        atom_mask[idx] = 1

    atom_mask[global_node_idx] = 1

    pharm_node_features = node_features * atom_mask[:, None]
    mask_matrix = np.outer(atom_mask, atom_mask)
    pharm_adjacency_matrix = adjacency_matrix * mask_matrix
    pharm_bond_features = bond_features * mask_matrix[:, :, None]

    return pharm_node_features, pharm_bond_features, pharm_adjacency_matrix


class PretrainMolecule:
    def __init__(self, mol, d_atom, d_edge, max_length, mask_ratio=0.15):
        self.smile = Chem.MolToSmiles(mol)
        self.node_features, self.bond_features, self.adjacency_matrix = load_data_from_mol(mol, d_atom, d_edge, max_length)
        self.num_nodes = np.count_nonzero(np.sum(np.abs(self.node_features), axis=-1)) 
        self.masked_node_features, self.masked_labels = mask_node_features(
            self.node_features,
            mask_ratio,
            include_global_node=False
        )
        self.scaffold_node_features, self.scaffold_bond_features, self.scaffold_adjacency_matrix = get_scaffold_mol(mol, self.node_features, self.bond_features, self.adjacency_matrix)
        self.scaffold_masked_node_features, self.scaffold_masked_labels = mask_node_features(
            self.scaffold_node_features,
            mask_ratio,
            include_global_node=False
        )
        self.fg_node_features, self.fg_bond_features, self.fg_adjacency_matrix = get_functional_group_mol(mol, self.node_features, self.bond_features, self.adjacency_matrix)
        self.fg_masked_node_features, self.fg_masked_labels = mask_node_features(
            self.fg_node_features,
            mask_ratio,
            include_global_node=False
        )
        self.pharm_node_features, self.pharm_bond_features, self.pharm_adjacency_matrix = get_pharmacophore_mol(mol, self.node_features, self.bond_features, self.adjacency_matrix)
        self.pharm_masked_node_features, self.pharm_masked_labels = mask_node_features(
            self.pharm_node_features,
            mask_ratio,
            include_global_node=False
        )


def pretrain_construct_dataset(mol_list, d_atom, d_edge, max_length):
    output = [PretrainMolecule(mol, d_atom, d_edge, max_length)
              for mol in tqdm(mol_list, total=len(mol_list))]
    return MolDataSet(output)

        
def pretrain_mol_collate_func(batch):
    num_nodes_list = []
    smile_list = []
    adjacency_matrices_list = [[] for _ in range(4)]
    node_features_list = [[] for _ in range(4)]
    bond_features_list = [[] for _ in range(4)]
    masked_labels_list = [[] for _ in range(4)]

    for molecule in batch:
        num_nodes_list.append(molecule.num_nodes)
        smile_list.append(molecule.smile)
        adjacency_matrices_list[0].append(molecule.adjacency_matrix)
        node_features_list[0].append(molecule.masked_node_features)
        bond_features_list[0].append(molecule.bond_features)
        masked_labels_list[0].append(molecule.masked_labels)
        adjacency_matrices_list[1].append(molecule.scaffold_adjacency_matrix)
        node_features_list[1].append(molecule.scaffold_masked_node_features)
        bond_features_list[1].append(molecule.scaffold_bond_features)
        masked_labels_list[1].append(molecule.scaffold_masked_labels)
        adjacency_matrices_list[2].append(molecule.fg_adjacency_matrix)
        node_features_list[2].append(molecule.fg_masked_node_features)
        bond_features_list[2].append(molecule.fg_bond_features)
        masked_labels_list[2].append(molecule.fg_masked_labels)
        adjacency_matrices_list[3].append(molecule.pharm_adjacency_matrix)
        node_features_list[3].append(molecule.pharm_masked_node_features)
        bond_features_list[3].append(molecule.pharm_bond_features)
        masked_labels_list[3].append(molecule.pharm_masked_labels)
    adjacency_matrices_list = [torch.from_numpy(np.array(adj_list)).float() for adj_list in adjacency_matrices_list]
    node_features_list = [torch.from_numpy(np.array(node_list)).float() for node_list in node_features_list]
    bond_features_list = [torch.from_numpy(np.array(bond_list)).float() for bond_list in bond_features_list]
    masked_labels_list = [torch.from_numpy(np.array(mask_list)).float() for mask_list in masked_labels_list]
    return [num_nodes_list,smile_list, adjacency_matrices_list, node_features_list, bond_features_list, masked_labels_list]

def pad_array(array, shape):
    padded_array = np.zeros(shape, dtype=float)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def construct_dataset(mol_list, label_list, d_atom, d_edge, max_length):
    output = [Molecule(mol, label, d_atom, d_edge, max_length)
              for (mol, label) in tqdm(zip(mol_list, label_list), total=len(mol_list))]
    return MolDataSet(output)

def mol_collate_func(batch):
    smile_list, adjacent_list, node_feature_list, bond_feature_list, label_list = [], [], [], [], []
    for molecule in batch:
        smile_list.append(molecule.smile)
        adjacent_list.append(molecule.adjacency_matrix)
        node_feature_list.append(molecule.node_features)
        bond_feature_list.append(molecule.bond_features)

        if isinstance(molecule.label, list):
            label_list.append(molecule.label)
        else:                                      
            label_list.append([molecule.label])

    return [smile_list] + [torch.from_numpy(np.array(features)).float() for features in (adjacent_list, node_feature_list, bond_feature_list, label_list)]


def construct_loader(mol_list, label_list, batch_size, d_atom, d_edge, max_length, shuffle=True):
    dataset = construct_dataset(mol_list, label_list, d_atom, d_edge, max_length + 1) 
    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=mol_collate_func, shuffle=shuffle,
                        drop_last=True, num_workers=0)
    return loader