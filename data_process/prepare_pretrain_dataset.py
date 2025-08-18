from rdkit import Chem
import pickle
import csv
path='../data/zinc15_250K.csv'
skip_smiles = set()
with open(path) as f:
    reader = csv.reader(f)
    next(reader)

    lines = []
    for line in reader:
        smiles = line[0]
        if smiles in skip_smiles:
            continue
        lines.append(line)

molecules=[]
data_list=[]
for smile in lines:
    mol = Chem.MolFromSmiles(smile[0])
    molecules.append(mol)

with open('../data/zinc15_250K/zinc15_250K.pickle', 'wb') as f:
     pickle.dump(molecules,f)