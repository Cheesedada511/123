import pandas as pd
import pickle as pkl
from rdkit import Chem

data = pd.read_csv('../data/bace.csv',sep=',')
data = data.fillna(-1.0)

data_smiles = data['smiles'].values.tolist()
data_labels = data.iloc[:, 1:].values.tolist()
# data_labels = data['measured log solubility in mols per litre'].values.tolist() # esol

data_san_mol, data_san_label = [], []
for smile, label in zip(data_smiles, data_labels):
    mol = Chem.MolFromSmiles(smile)
    try:
        Chem.SanitizeMol(mol)
    except:
        continue

    if mol.GetNumAtoms() >= 200:
        continue

    data_san_mol.append(mol)
    data_san_label.append(label)

with open('../data/bace/bace.pickle', 'wb') as fw:
    pkl.dump([data_san_mol, data_san_label],fw)