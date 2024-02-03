import numpy as np
import pandas as pd
import os 
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit import DataStructs
import mordred
from mordred import Calculator, descriptors
from tqdm import tqdm




def customTanimoto(a, b):
    return np.sum(a*b)/(np.sum(a**2) + np.sum(b**2) - np.sum(a*b))


def getMODIindex(df, fps, fp = 'rdkit'):
    
    """
    To calculate the Modi-index
     
    For Binary classification

    
    MODI = 1/2 * ∑ [i=1 to 2] (N[i][same] / N[i][total])
    
    Parameters
    ----------
    df : Dataframe 
        containing the following columns-
            SMILES - SMILE molecule string.
            Activity - respective classification.
            
    fps : List of Fingerprints
    
    fp : rdkit Tanimoto similarity index 

    
    Returns
    -------
    Modi index
        integer value.

    """
    modified_df = df.copy()
    modified_df['fps'] = fps

    Modi = 0
    for activity in modified_df['Activity'].unique():
        temp_df = modified_df[modified_df['Activity'] == activity]
        Ni_same = 0
        Ni_total = 0
        for i in range(temp_df.shape[0]):
            sim = []
            if temp_df['fps'].iloc[i] is None:
                continue
            for j in range(modified_df.shape[0]):
                if modified_df['fps'].iloc[j] is None:
                    continue
                if temp_df['SMILES'].iloc[i] != modified_df['SMILES'].iloc[j]:
                    if fp == 'rdkit':
                        sim.append((DataStructs.TanimotoSimilarity(temp_df['fps'].iloc[i], modified_df['fps'].iloc[j]),
                                    temp_df['Activity'].iloc[i] == modified_df['Activity'].iloc[j]))
                    else:
                        sim.append((customTanimoto(temp_df['fps'].iloc[i], modified_df['fps'].iloc[j]), 
                                    temp_df['Activity'].iloc[i] == modified_df['Activity'].iloc[j]))
            
            max_element = max(sim, key = lambda x: x[0])

            if max_element[1]:
                Ni_same += 1
            Ni_total += 1

        Modi += Ni_same/Ni_total

    Modi = Modi/len(modified_df['Activity'].unique())

    return Modi





def maccs_fp(smiles):
    """
    MACCS fringerprint generator from a SMILES molecule structure.

    Parameters
    ----------
    smiles : List [str]
        containing SMILES structures.

    
    Returns
    -------
    List
        List of fingerprint of each SMILE structure.

    """
    
    fps = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            fps.append(None)
        else:
            fps.append(MACCSkeys.GenMACCSKeys(mol))

    assert len(fps) == len(smiles)
    return fps



 
def morgen_fp(smiles):    
    """
    Morgen fingerprint generator from SMILES structure.

    Parameters
    ----------
    smiles : List [str]
        The SMILES string defining the molecule.

   
    Returns
    -------
    List
        List of fingerprint of each SMILE structure.

    """
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            fps.append(None)
        else:
            fps.append(fpg.GetFingerprint(mol))

    assert len(fps) == len(smiles)
    return fps
    
    

def mordred_fp(smiles):
    """
    Mordred fringerpint calculator using all the descriptor using smile structure .

    Parameters
    ----------
    smiles : List [str]
        It is a list of SMILES structures
     
    Returns
    -------
    List
        List of fingerprint of each SMILE structure.

    """
    calc = Calculator(descriptors, ignore_3D=True)
    fps = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            fps.append(None)
        else:
            # if calc(mol)
            fps.append(np.array(calc(mol)))
            

    assert len(fps) == len(smiles)
    return fps
    
    
def valid_indices(smiles):
    """
    Parameter
    ---------
    Smiles: List [str]
        Takes Raw smiles structure as input 
    

    Output:
    -------
        Returns a List of indices corresponding to smiles structure having valid molecular structure/ entry
    """
    valid_idx = []
    for i,x in enumerate(smiles):
        y = Chem.MolFromSmiles(x)
        
        if y is not None:
            valid_idx.append(i)
    
    return valid_idx






class mordredWrapper:
    def __init__(self, smiles_array) -> None:
        self.calc = Calculator(descriptors, ignore_3D=True)

        mol_list = []
        for smiles in smiles_array:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_list.append(mol)

        df_mordred = self.calc.pandas(mol_list)
        truth_map = df_mordred.applymap(lambda x : not isinstance(x, mordred.error.MissingValueBase))
        truth_series = truth_map.all(axis=0)
        self.mask = truth_series.to_numpy()

    def get_fingerprints(self, smiles_array, labels):
        fps = []
        y = []
        for smiles, label in zip(smiles_array, labels):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                pass
            else:
                fps.append(np.array(self.calc(mol))[self.mask])
                y.append(label)

        assert len(fps) == len(y)
        
        return np.array(fps), np.array(y)

            