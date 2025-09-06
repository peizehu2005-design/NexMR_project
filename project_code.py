# corrected_graph_builder.py
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Callable, List
import inspect

# ---------------- helpers ----------------
def _hyb_to_number(atom: Chem.rdchem.Atom) -> float:
    """Map common hybridizations to numbers; returns 0.0 for others."""
    hyb = atom.GetHybridization()
    mapping = {
        Chem.rdchem.HybridizationType.SP: 1.0,
        Chem.rdchem.HybridizationType.SP2: 2.0,
        Chem.rdchem.HybridizationType.SP3: 3.0
    }
    return float(mapping.get(hyb, 0.0))

def default_attention_fn(angle_rad: float) -> float:
    sigma = np.pi / 4.0
    return float(np.exp(-0.5 * (angle_rad / sigma) ** 2))

def _call_attention_fn(att_fn: Callable, angle: float, e_idx: int, f_idx: int) -> float:
    """Call user attention fn with flexible signatures."""
    if att_fn is None:
        return default_attention_fn(angle)
    try:
        nparams = len(inspect.signature(att_fn).parameters)
        if nparams >= 3:
            return float(att_fn(angle, e_idx, f_idx))
        elif nparams == 2:
            return float(att_fn(angle, e_idx))
        elif nparams == 1:
            return float(att_fn(angle))
        else:
            return float(att_fn())
    except Exception:
        return default_attention_fn(angle)

def bond_neighbors(mol: Chem.Mol, bond_idx: int) -> List[int]:
    """Return indices of bonds that share an atom with bond_idx (excluding bond_idx)."""
    bond = mol.GetBondWithIdx(bond_idx)
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    neigh = set()
    for atom_idx in (i, j):           # tuple iteration: first i, then j
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetBonds():
            if nb.GetIdx() != bond_idx:
                neigh.add(nb.GetIdx())
    return sorted(neigh)

def compute_angle_between_bonds(bond_a_ends: Tuple[int, int],
                                bond_b_ends: Tuple[int, int],
                                pos: np.ndarray) -> float:
    """Angle (radians) between bonds at their shared atom. pos is (N_atoms,3)."""
    setA, setB = set(bond_a_ends), set(bond_b_ends)
    shared = setA & setB
    if len(shared) != 1:
        raise ValueError("Bonds must share exactly one atom")
    s = next(iter(shared))
    other_a = (setA - {s}).pop()
    other_b = (setB - {s}).pop()

    v1 = pos[other_a] - pos[s]
    v2 = pos[other_b] - pos[s]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cosang))

def _is_bond_rotatable(bond: Chem.rdchem.Bond) -> bool:
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    return (a.GetDegree() > 1) and (b.GetDegree() > 1)

def _hydrogen_exchange_rule(bond: Chem.rdchem.Bond) -> bool:
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    def is_labile(atom):
        return atom.GetAtomicNum() in (7, 8, 16)  # N,O,S
    if a.GetAtomicNum() == 1 and is_labile(b): return True
    if b.GetAtomicNum() == 1 and is_labile(a): return True
    return False

# ---------------- atom & bond featurizers ----------------
def get_atom_features(atom: Chem.rdchem.Atom,
                      mulliken_charges: List[float] = None,
                      loewdin_charges: List[float] = None) -> List[float]:
    idx = atom.GetIdx()
    Z = float(atom.GetAtomicNum())
    mull = float(mulliken_charges[idx]) if mulliken_charges is not None else 0.0
    loe = float(loewdin_charges[idx]) if loewdin_charges is not None else 0.0
    hyb = _hyb_to_number(atom)
    return [Z, mull, loe, hyb]

def get_bond_numeric_features(bond: Chem.rdchem.Bond, conf: Chem.rdchem.Conformer) -> Tuple[List[float], Tuple[int,int]]:
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    pi = np.array(conf.GetAtomPosition(i))
    pj = np.array(conf.GetAtomPosition(j))
    blen = float(np.linalg.norm(pi - pj))
    btype = float(bond.GetBondTypeAsDouble())
    rot = 1.0 if _is_bond_rotatable(bond) else 0.0
    h_ex = 1.0 if _hydrogen_exchange_rule(bond) else 0.0
    return [blen, btype, rot, h_ex], (i, j)

# ---------------- compute per-bond attention arrays ----------------
def compute_all_bond_attentions(mol: Chem.Mol,
                                bond_endpoints: List[Tuple[int,int]],
                                attention_fn: Callable = None) -> Tuple[List[List[int]], List[torch.Tensor]]:
    """Return (bond_neighbors_list, bond_attn_tensor_list)."""
    # ensure coordinates exist
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=float)

    B = len(bond_endpoints)
    bond_neighbors_list: List[List[int]] = []
    bond_attn_list: List[torch.Tensor] = []
    for bidx in range(B):
        neighs = bond_neighbors(mol, bidx)
        bond_neighbors_list.append(neighs)
        att_vals = []
        for fidx in neighs:
            angle = compute_angle_between_bonds(bond_endpoints[bidx], bond_endpoints[fidx], pos)
            a_val = _call_attention_fn(attention_fn, angle, bidx, fidx)
            att_vals.append(a_val)
        bond_attn_list.append(torch.tensor(att_vals, dtype=torch.float))  # possibly empty tensor
    return bond_neighbors_list, bond_attn_list

# ---------------- build Data ----------------
def sdf_to_graph(sdf_path: str,
                 mulliken_charges: List[float] = None,
                 loewdin_charges: List[float] = None,
                 attention_fn: Callable = None) -> Data:
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError("Invalid SDF")
    # ensure coords
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

    conf = mol.GetConformer()

    # nodes
    node_feats = [get_atom_features(a, mulliken_charges, loewdin_charges) for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    # bonds (unique)
    bond_numeric_list = []
    bond_endpoints = []
    for bond in mol.GetBonds():
        feat, ends = get_bond_numeric_features(bond, conf)
        bond_numeric_list.append(feat)
        bond_endpoints.append(ends)

    # directional edges + edge_attr (numeric only)
    edge_pairs = []
    edge_attrs = []
    for bidx, (i,j) in enumerate(bond_endpoints):
        edge_pairs.append([i,j])
        edge_pairs.append([j,i])
        edge_attrs.append(bond_numeric_list[bidx])
        edge_attrs.append(bond_numeric_list[bidx])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # positions
    pos = torch.tensor([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=torch.float)

    # compute bond-bond attentions separately
    bond_neighbors_list, bond_attn_list = compute_all_bond_attentions(mol, bond_endpoints, attention_fn)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    # attach bond-centric metadata (variable-length)
    data.bond_endpoints = bond_endpoints
    data.bond_neighbors = bond_neighbors_list
    data.bond_attn = bond_attn_list
    data.bond_numeric = torch.tensor(bond_numeric_list, dtype=torch.float)

    return data
    
def two_mol_graph(sdf1, sdf2, mulliken1=None, loewdin1=None, mulliken2=None, loewdin2=None, att_fn=None):
    g1 = sdf_to_graph(sdf1, mulliken1, loewdin1, att_fn)
    g2 = sdf_to_graph(sdf2, mulliken2, loewdin2, att_fn)
    
    # Offset node indices for second molecule
    offset = g1.x.size(0)
    g2.edge_index += offset
    
    # Merge graphs
    x = torch.cat([g1.x, g2.x], dim=0)
    pos = torch.cat([g1.pos, g2.pos], dim=0)
    edge_index = torch.cat([g1.edge_index, g2.edge_index], dim=1)
    edge_attr = torch.cat([g1.edge_attr, g2.edge_attr], dim=0)
    
    # Add COM edge
    com1 = g1.pos.mean(dim=0, keepdim=True)
    com2 = g2.pos.mean(dim=0, keepdim=True)
    com1_idx, com2_idx = offset + g2.x.size(0), offset + g2.x.size(0) + 1
    
    # Add virtual nodes for COMs
    x = torch.cat([x, torch.zeros((2, x.size(1)))], dim=0)
    pos = torch.cat([pos, com1, com2], dim=0)
    
    edge_index = torch.cat([edge_index,
                            torch.tensor([[com1_idx, com2_idx],
                                          [com2_idx, com1_idx]], dtype=torch.long)], dim=1)
    
    # No attributes for COM edge (placeholder: could add distance, etc.)
    edge_attr = torch.cat([edge_attr, torch.zeros((2, edge_attr.size(1)))], dim=0)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Example usage
graph = two_mol_graph("mol1.sdf", "mol2.sdf")
print(graph)
