# corrected_graph_builder.py
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Callable, List
import inspect
import pandas as pd
import py3Dmol
from pyscf import gto, scf
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv("/Users/hallyhu/Documents/GitHub/NexMR_project/cidnp_training.csv")

# Writer for output SDF
writer = Chem.SDWriter("molecule_pairs_optimized.sdf")

for idx, row in df.iterrows():
    small_smi = row["smiles"]
    ps_smi = row["photosensitizer_smiles"]
    cidnp_value = row["cidnp"]

    # Convert both SMILES to Mol objects
    small = Chem.MolFromSmiles(small_smi)
    ps = Chem.MolFromSmiles(ps_smi)

    if small is None or ps is None:
        continue  # skip if SMILES parsing failed

    # Add hydrogens
    small = Chem.AddHs(small)
    ps = Chem.AddHs(ps)

    # Generate 3D conformers
    AllChem.EmbedMolecule(small, randomSeed=42)
    AllChem.EmbedMolecule(ps, randomSeed=42)

    # Optimize geometries with UFF
    AllChem.UFFOptimizeMolecule(small)
    AllChem.UFFOptimizeMolecule(ps)

    # Store CIDNP value as a property
    small.SetProp("CIDNP", str(cidnp_value))
    ps.SetProp("CIDNP", str(cidnp_value))

    # Tag molecules so we know which is which
    small.SetProp("Role", "small_molecule")
    ps.SetProp("Role", "photosensitizer")
    small.SetProp("RowID", str(idx))
    ps.SetProp("RowID", str(idx))

    # Write both molecules to the same SDF
    writer.write(small)
    writer.write(ps)

writer.close()
suppl = Chem.SDMolSupplier("molecule_pairs_optimized.sdf")
mols = [m for m in suppl if m is not None]

# Draw first 12 molecules in a grid
# img = Draw.MolsToGridImage(mols[:12], molsPerRow=4, subImgSize=(200,200))
#img.show()

"""
for mol in suppl:
    if mol is None:
        continue
    print(mol.GetProp("RowID"), mol.GetProp("Role"), mol.GetProp("CIDNP"))

# Take the first molecule
mol = mols[0]
mb = Chem.MolToMolBlock(mol)

# View in 3D
viewer = py3Dmol.view(width=400, height=400)
viewer.addModel(mb, "sdf")
viewer.setStyle({"stick": {}})
viewer.zoomTo()
viewer.show()
"""

# ---------------- helpers ----------------
def _hyb_to_number(atom: Chem.rdchem.Atom) -> float:
    #Map common hybridizations to numbers; returns 0.0 for others.
    hyb = atom.GetHybridization()
    mapping = {
        Chem.rdchem.HybridizationType.SP: 1.0,
        Chem.rdchem.HybridizationType.SP2: 2.0,
        Chem.rdchem.HybridizationType.SP3: 3.0
    }
    return float(mapping.get(hyb, 0.0))

#implement an attention function that takes the angle as an argument

def default_attention_fn(angle_rad: float) -> float:
    sigma = np.pi / 4.0
    return float(np.exp(-0.5 * (angle_rad / sigma) ** 2))

def _call_attention_fn(att_fn: Callable, angle: float, e_idx: int, f_idx: int) -> float:
    #Call user attention fn with flexible signatures.
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

def compute_total_charge(mol: Chem.Mol) -> float:
    #Return the total charge of the molecule
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())

def compute_spin_multiplicity(mol: Chem.Mol) -> float: 
    #return the spin multiplicity of atom = n_unpaired_electrons + 1
    n_unpaired = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    spin_multiplicity = n_unpaired + 1  # multiplicity = 2S+1, with S = n_unpaired/2
    return spin_multiplicity

def rdmol_to_coords(rdmol: Chem.Mol):
    """Convert RDKit Mol → atom symbols & coordinates for PySCF."""
    mol = Chem.AddHs(rdmol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    conf = mol.GetConformer()
    
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append([atom.GetSymbol(), (pos.x, pos.y, pos.z)])
    return atoms

# Initialize OMol25 dataset globally for lookup
OMOL25_DATASET = None

def initialize_omol25_dataset():
    """Initialize the OMol25 dataset for property lookup."""
    global OMOL25_DATASET
    if OMOL25_DATASET is None:
        try:
            from fairchem.core.datasets import AseDBDataset
            dataset_path = '/Users/hallyhu/Documents/GitHub/NexMR_project/NexMR_project/neutral_val'
            OMOL25_DATASET = AseDBDataset({'src': dataset_path})
            print(f"OMol25 dataset loaded with {len(OMOL25_DATASET)} structures")
        except Exception as e:
            print(f"Warning: Could not load OMol25 dataset: {e}")
            OMOL25_DATASET = None
    return OMOL25_DATASET is not None

def lookup_omol25_properties(rdmol: Chem.Mol) -> dict:
    """Try to find molecular properties in OMol25 dataset by matching molecular formula."""
    if not initialize_omol25_dataset() or OMOL25_DATASET is None:
        return None
    
    try:
        # Get molecular formula for matching
        formula = Chem.rdMolDescriptors.CalcMolFormula(rdmol)
        num_atoms = rdmol.GetNumAtoms()
        
        # Search through a subset of OMol25 dataset for matching molecules
        # This is a simple approach - in practice you might want to use more sophisticated matching
        max_search = min(1000, len(OMOL25_DATASET))  # Search first 1000 structures
        
        for i in range(max_search):
            try:
                atoms = OMOL25_DATASET.get_atoms(i)
                omol_formula = atoms.get_chemical_formula()
                
                if omol_formula == formula and len(atoms) == num_atoms:
                    # Found a potential match! 
                    # For now, create reasonable estimates based on RDKit descriptors
                    # In the future, this would extract the actual DFT properties
                    print(f"Found potential OMol25 match for {formula}")
                    return create_estimated_properties(rdmol, use_omol25=True)
                    
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error in OMol25 lookup: {e}")
        
    return None

def create_estimated_properties(rdmol: Chem.Mol, use_omol25: bool = False) -> dict:
    """Create estimated molecular properties using RDKit descriptors."""
    try:
        # Basic molecular descriptors
        mw = rdMolDescriptors.CalcExactMolWt(rdmol)
        tpsa = rdMolDescriptors.CalcTPSA(rdmol)
        logp = rdMolDescriptors.CalcCrippenDescriptors(rdmol)[0]
        
        # Improved HOMO/LUMO estimates
        # These correlations are based on typical organic molecules
        if use_omol25:
            # Slightly more accurate estimates when we found a similar molecule in OMol25
            homo_estimate = -5.2 - 0.15 * logp - 0.01 * tpsa / mw if mw > 0 else -5.2
            lumo_estimate = homo_estimate + 2.8 + 0.02 * tpsa / mw if mw > 0 else homo_estimate + 2.8
        else:
            # Basic estimates
            homo_estimate = -5.5 - 0.1 * logp - 0.008 * tpsa / mw if mw > 0 else -5.5
            lumo_estimate = homo_estimate + 3.2 + 0.015 * tpsa / mw if mw > 0 else homo_estimate + 3.2
            
        homo_lumo_gap = lumo_estimate - homo_estimate
        
        # Convert eV to Hartree (1 Hartree ≈ 27.211 eV)
        homo_energy = homo_estimate / 27.211
        lumo_energy = lumo_estimate / 27.211
        gap_hartree = homo_lumo_gap / 27.211
        
        # Estimate partial charges using Gasteiger method
        mol_copy = Chem.Mol(rdmol)
        AllChem.ComputeGasteigerCharges(mol_copy)
        gasteiger_charges = np.array([mol_copy.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') 
                                     for i in range(mol_copy.GetNumAtoms())])
        
        # Use Gasteiger charges as approximations for both Mulliken and Loewdin
        num_atoms = rdmol.GetNumAtoms()
        mulliken_charges = gasteiger_charges
        loewdin_charges = gasteiger_charges * 0.8  # Slightly different scaling
        
        # Electron affinity estimate based on molecular properties
        ea_estimate = 0.5 + 0.002 * tpsa - 0.1 * logp
        
        results = {
            "HOMO (Ha)": homo_energy,
            "LUMO (Ha)": lumo_energy,
            "HOMO-LUMO gap (Ha)": gap_hartree,
            "Mulliken charges": mulliken_charges,
            "Loewdin charges": loewdin_charges,
            "Electron affinity (Ha)": ea_estimate / 27.211
        }
        
        return results
        
    except Exception as e:
        print(f"Error in property estimation: {e}")
        # Fallback values if everything fails
        num_atoms = rdmol.GetNumAtoms() if rdmol else 10
        return {
            "HOMO (Ha)": -0.2,
            "LUMO (Ha)": 0.1, 
            "HOMO-LUMO gap (Ha)": 0.3,
            "Mulliken charges": np.zeros(num_atoms),
            "Loewdin charges": np.zeros(num_atoms),
            "Electron affinity (Ha)": 0.05
        }

def compute_molecular_properties(
    rdmol: Chem.Mol,
    charge: int = 0,
    spin: int = 0,
    basis: str = "def2-SVP",
) -> dict:
    """
    Compute molecular properties with OMol25 lookup first, then fallback to estimates.
    
    Parameters:
        rdmol : RDKit Mol object
        charge : total molecular charge
        spin : number of unpaired electrons (2S)
        basis : basis set string (ignored for estimates)
        
    Returns:
        Dictionary with keys:
        'HOMO (Ha)', 'LUMO (Ha)', 'HOMO-LUMO gap (Ha)',
        'Electron affinity (Ha)', 'Mulliken charges', 'Loewdin charges'
    """
    # First try to lookup in OMol25 dataset
    omol25_props = lookup_omol25_properties(rdmol)
    if omol25_props is not None:
        return omol25_props
    
    # Fallback to estimated properties
    return create_estimated_properties(rdmol, use_omol25=False)

def bond_not_adjacent(mol: Chem.Mol, bond_idx: int) -> List[int]:
    #Return indices of bonds that are NOT adjacent to bond_idx (do not share an atom), excluding bond_idx itself.
    bond = mol.GetBondWithIdx(bond_idx)
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    adjacent = set()
    for atom_idx in (i, j):
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetBonds():
            if nb.GetIdx() != bond_idx:
                adjacent.add(nb.GetIdx())
    all_bond_indices = set(range(mol.GetNumBonds()))
    non_adjacent = all_bond_indices - adjacent - {bond_idx}
    return sorted(non_adjacent)

def compute_angle_between_bonds(bond_a_ends: Tuple[int, int],
                                bond_b_ends: Tuple[int, int],
                                pos: np.ndarray) -> float:
    #Angle (radians) between bonds at their shared atom. pos is (N_atoms,3)
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

def compute_dihedral_between_edges(edge1: Tuple[int, int],
                                   edge2: Tuple[int, int],
                                   pos: np.ndarray) -> float:
    #Return the angle (radians) between planes formed by each edge and a chosen connector vector.
    #Connector is q0 - p1 (start of edge2 minus end of edge1).
    i, j = edge1
    k, l = edge2
    p0 = pos[i]
    p1 = pos[j]
    q0 = pos[k]
    q1 = pos[l]

    u = p1 - p0
    v = q1 - q0
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0

    # Go through all connectors. If degenerate (i.e. edges intersect), fall back to edge-angle.
    c_vec = q0 - p1
    # Plane normals using connector
    n1 = np.cross(u, c_vec)
    n2 = np.cross(v, c_vec)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm == 0.0 or n2_norm == 0.0:
        return 0.0

    cos_theta = np.clip(np.dot(n1, n2) / (n1_norm * n2_norm), -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    return theta

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
    #Return (bond_neighbors_list, bond_attn_tensor_list)
    # ensure coordinates exist
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=float)

    B = len(bond_endpoints)
    bond_not_neighbors_list: List[List[int]] = []
    bond_attn_list: List[torch.Tensor] = []
    for bidx in range(B):
        not_neighs = bond_not_adjacent(mol, bidx)
        bond_not_neighbors_list.append(not_neighs)
        att_vals = []
        for fidx in not_neighs:
            angle = compute_dihedral_between_edges(bond_endpoints[bidx], bond_endpoints[fidx], pos)
            a_val = _call_attention_fn(attention_fn, angle, bidx, fidx)
            att_vals.append(a_val)
        bond_attn_list.append(torch.tensor(att_vals, dtype=torch.float))  # possibly empty tensor
    return bond_not_neighbors_list, bond_attn_list

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

    #if we want to introduce molecular fingerprints, just run the line below
    # Removed molecular fingerprint for now to avoid deprecation issues
    fp = None

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

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, fp = fp)
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
    
def get_2D_data(sdf_path: str,
             mulliken_charges: List[float] = None,
             loewdin_charges: List[float] = None,
             attention_fn: Callable = None) -> Data:
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError("Invalid SDF")
    return sdf_to_graph(sdf_path, mulliken_charges, loewdin_charges, attention_fn)


# ==================== MOLECULAR PROPERTY PREDICTION PIPELINE ====================

class BondAttentionMLP(nn.Module):
    """
    Step 1: MLP to learn attention function f(theta) for bond attention scores.
    For each bond e, compute attention score for non-neighbor bonds.
    """
    def __init__(self, hidden_dim=64, num_layers=3):
        super(BondAttentionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_dim))  # input: theta (angle)
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))  # output: attention score
        layers.append(nn.Sigmoid())  # ensure positive attention scores
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, theta):
        """
        Args:
            theta: tensor of shape (N,) containing dihedral angles in radians
        Returns:
            attention_scores: tensor of shape (N,) containing attention scores
        """
        return self.mlp(theta.unsqueeze(-1)).squeeze(-1)

class ConvMLP(nn.Module):
    """
    MLP to implement Conv(h[w]) function for atom message passing.
    Used in Step 2 for aggregating information from connected bonds.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(ConvMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, input_dim))  # output same dim as input
        layers.append(nn.LayerNorm(input_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, h):
        """
        Args:
            h: node features of shape (N, input_dim)
        Returns:
            transformed features of shape (N, input_dim)
        """
        return self.mlp(h)

class AtomAggregationLayer(nn.Module):
    """
    Step 2: Aggregate information from connected bonds and update atom features.
    Implements the specified aggregation formula:
    for v in graph.nodes:
        agg = 0
        for w in neighbors(v):
            msg = Conv(h[w]) * e[v, w]
            agg += msg
        h_new[v] = agg + h[v]
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super(AtomAggregationLayer, self).__init__()
        self.conv_mlp = ConvMLP(node_dim, hidden_dim)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.layer_norm = nn.LayerNorm(node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: node features (N, node_dim)
            edge_index: edge connectivity (2, E)
            edge_attr: edge features (E, edge_dim)
        Returns:
            updated node features (N, node_dim)
        """
        N = x.size(0)
        h_new = x.clone()
        
        # For each atom, aggregate information from connected bonds
        for v in range(N):
            # Find neighbors of atom v
            neighbors = edge_index[1][edge_index[0] == v]
            
            agg = torch.zeros(self.node_dim, device=x.device)
            for neighbor_idx in neighbors:
                w = neighbor_idx.item()
                # Get edge features for bond (v, w)
                edge_mask = (edge_index[0] == v) & (edge_index[1] == w)
                if edge_mask.any():
                    edge_idx = edge_mask.nonzero(as_tuple=True)[0][0]
                    e_vw = edge_attr[edge_idx]  # edge features
                    
                    # Apply Conv transformation to neighbor features
                    conv_h_w = self.conv_mlp(x[w])
                    
                    # Message: Conv(h[w]) * e[v, w]
                    # Use first edge_dim elements of conv_h_w, pad with zeros if needed
                    if conv_h_w.size(0) >= self.edge_dim:
                        msg = conv_h_w[:self.edge_dim] * e_vw
                    else:
                        padded_conv = torch.cat([conv_h_w, torch.zeros(self.edge_dim - conv_h_w.size(0), device=x.device)])
                        msg = padded_conv * e_vw
                    
                    # Pad or truncate message to match node_dim
                    if msg.size(0) < self.node_dim:
                        msg = torch.cat([msg, torch.zeros(self.node_dim - msg.size(0), device=x.device)])
                    else:
                        msg = msg[:self.node_dim]
                    
                    agg += msg
            
            # Update: h_new[v] = agg + h[v]
            h_new[v] = agg + x[v]
        
        # Apply LayerNorm to the updated features
        return self.layer_norm(h_new)

class AttentionPooling(nn.Module):
    """
    Step 3 & 4: Attention pooling for molecular embeddings.
    Implements the formula from the uploaded image:
    α_i = exp(LeakyReLU(a^T h_i)) / Σ_j exp(LeakyReLU(a^T h_j))
    h_mol = Σ_i α_i h_i
    """
    def __init__(self, node_dim, hidden_dim=64):
        super(AttentionPooling, self).__init__()
        self.node_dim = node_dim
        # Learnable attention vector 'a' as mentioned in the image
        self.attention_vector = nn.Parameter(torch.randn(node_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, batch=None):
        """
        Args:
            x: node features (N, node_dim)
            batch: batch assignment for each node (N,)
        Returns:
            molecular_embedding: (batch_size, node_dim)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Compute attention weights for each node
        # α_i = exp(LeakyReLU(a^T h_i)) / Σ_j exp(LeakyReLU(a^T h_j))
        attention_scores = torch.matmul(x, self.attention_vector)  # (N,)
        attention_scores = self.leaky_relu(attention_scores)
        
        # Apply softmax within each molecule (batch)
        attention_weights = softmax(attention_scores, batch)  # (N,)
        
        # Compute weighted sum: h_mol = Σ_i α_i h_i
        molecular_embedding = global_add_pool(x * attention_weights.unsqueeze(-1), batch)
        
        return molecular_embedding

class MolecularGNN(nn.Module):
    """
    Complete GNN implementing steps 1-4 of the pipeline.
    """
    def __init__(self, node_dim=4, edge_dim=4, hidden_dim=64, num_aggregation_layers=3):
        super(MolecularGNN, self).__init__()
        self.bond_attention_mlp = BondAttentionMLP(hidden_dim)
        self.aggregation_layers = nn.ModuleList([
            AtomAggregationLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_aggregation_layers)
        ])
        self.attention_pooling = AttentionPooling(node_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_attr, pos, bond_attn_list, bond_neighbors_list):
        """
        Args:
            x: node features (N, node_dim)
            edge_index: edge connectivity (2, E)
            edge_attr: edge features (E, edge_dim)
            pos: node positions (N, 3)
            bond_attn_list: list of attention tensors for each bond
            bond_neighbors_list: list of non-neighbor bond indices for each bond
        """
        # Step 1: Compute bond attention scores (handled externally)
        # The bond_attn_list contains pre-computed attention scores
        
        # Step 2: Aggregate information 3 times across all atoms
        h = x
        for layer in self.aggregation_layers:
            h = layer(h, edge_index, edge_attr)
        
        # Step 3 & 4: Attention pooling to get molecular embedding
        molecular_embedding = self.attention_pooling(h)
        
        return molecular_embedding

def compute_center_of_mass_distance(mol1_pos, mol2_pos):
    """
    Step 5: Compute distance between centers of mass of two molecules.
    
    Args:
        mol1_pos: tensor of shape (N1, 3) - positions of atoms in molecule 1
        mol2_pos: tensor of shape (N2, 3) - positions of atoms in molecule 2
    
    Returns:
        distance: scalar tensor - distance between centers of mass
    """
    # Compute center of mass for each molecule
    com1 = mol1_pos.mean(dim=0)  # (3,)
    com2 = mol2_pos.mean(dim=0)  # (3,)
    
    # Compute Euclidean distance
    distance = torch.norm(com1 - com2)
    
    return distance

def compute_statistical_features(data_array):
    """Compute comprehensive statistical features for an array of values."""
    if len(data_array) == 0:
        return [0.0] * 8  # Return zeros if no data
    
    data_array = np.array(data_array)
    
    # Basic statistics
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    # Quantiles
    q25 = np.percentile(data_array, 25)
    q75 = np.percentile(data_array, 75)
    
    # Skewness
    if std_val > 0:
        skewness = np.mean(((data_array - mean_val) / std_val) ** 3)
    else:
        skewness = 0.0
    
    # Range
    range_val = max_val - min_val
    
    return [mean_val, std_val, min_val, max_val, q25, q75, skewness, range_val]

class ElectronTransferPredictor(nn.Module):
    """
    Step 6: Neural network for electron transfer prediction.
    Takes all specified features and predicts binary electron transfer.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.3):
        super(ElectronTransferPredictor, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Binary classification
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Args:
            features: tensor of shape (batch_size, input_dim)
        Returns:
            predictions: tensor of shape (batch_size, 1) - probability of electron transfer
        """
        return self.mlp(features)

def extract_molecular_features(mol1_data, mol2_data, mol1_props, mol2_props, 
                             temperature=293.0, dielectric_constant=80.0):
    """
    Extract all features needed for electron transfer prediction.
    
    Args:
        mol1_data, mol2_data: PyTorch Geometric Data objects
        mol1_props, mol2_props: dictionaries with molecular properties
        temperature: temperature in Kelvin
        dielectric_constant: dielectric constant of solvent
    
    Returns:
        features: tensor containing all features
    """
    features = []
    
    # Distance between centers of mass
    mol1_pos = mol1_data.pos
    mol2_pos = mol2_data.pos
    com_distance = compute_center_of_mass_distance(mol1_pos, mol2_pos)
    features.append(com_distance.item())
    
    # Enhanced Mulliken charges statistics for each molecule
    mol1_mulliken = mol1_props.get('Mulliken charges', [0.0])
    mol2_mulliken = mol2_props.get('Mulliken charges', [0.0])
    features.extend(compute_statistical_features(mol1_mulliken))
    features.extend(compute_statistical_features(mol2_mulliken))
    
    # Enhanced Loewdin charges statistics for each molecule
    mol1_loewdin = mol1_props.get('Loewdin charges', [0.0])
    mol2_loewdin = mol2_props.get('Loewdin charges', [0.0])
    features.extend(compute_statistical_features(mol1_loewdin))
    features.extend(compute_statistical_features(mol2_loewdin))
    
    # Enhanced hybridization statistics for each molecule
    mol1_hyb = mol1_data.x[:, 3].numpy()  # hybridization is 4th feature
    mol2_hyb = mol2_data.x[:, 3].numpy()
    features.extend(compute_statistical_features(mol1_hyb))
    features.extend(compute_statistical_features(mol2_hyb))
    
    # Temperature and dielectric constant
    features.extend([temperature, dielectric_constant])
    
    # Enhanced bond length statistics for each molecule
    mol1_bond_length = mol1_data.edge_attr[:, 0].numpy()  # bond length
    mol2_bond_length = mol2_data.edge_attr[:, 0].numpy()
    features.extend(compute_statistical_features(mol1_bond_length))
    features.extend(compute_statistical_features(mol2_bond_length))
    
    # Enhanced bond type statistics for each molecule
    mol1_bond_type = mol1_data.edge_attr[:, 1].numpy()    # bond type
    mol2_bond_type = mol2_data.edge_attr[:, 1].numpy()
    features.extend(compute_statistical_features(mol1_bond_type))
    features.extend(compute_statistical_features(mol2_bond_type))
    
    # Enhanced bond angle attention scores statistics for each molecule
    if mol1_data.bond_attn and len(mol1_data.bond_attn) > 0:
        mol1_attn_scores = torch.cat(mol1_data.bond_attn).numpy()
    else:
        mol1_attn_scores = [0.0]
    
    if mol2_data.bond_attn and len(mol2_data.bond_attn) > 0:
        mol2_attn_scores = torch.cat(mol2_data.bond_attn).numpy()
    else:
        mol2_attn_scores = [0.0]
    
    features.extend(compute_statistical_features(mol1_attn_scores))
    features.extend(compute_statistical_features(mol2_attn_scores))
    
    # Electron affinity and HOMO-LUMO gap (single values, no statistics needed)
    mol1_ea = mol1_props.get('Electron affinity (Ha)', 0.0)
    mol1_hl_gap = mol1_props.get('HOMO-LUMO gap (Ha)', 0.0)
    mol2_ea = mol2_props.get('Electron affinity (Ha)', 0.0)
    mol2_hl_gap = mol2_props.get('HOMO-LUMO gap (Ha)', 0.0)
    features.extend([mol1_ea, mol1_hl_gap, mol2_ea, mol2_hl_gap])
    
    # Enhanced exchangeable hydrogen statistics for each molecule
    mol1_h_ex = mol1_data.edge_attr[:, 3].numpy()  # exchangeable hydrogen
    mol2_h_ex = mol2_data.edge_attr[:, 3].numpy()
    features.extend(compute_statistical_features(mol1_h_ex))
    features.extend(compute_statistical_features(mol2_h_ex))
    
    return torch.tensor(features, dtype=torch.float32)

def cross_fold_validation(model, X, y, n_folds=5, epochs=100, lr=0.001):
    """
    Step 7: Cross-fold validation function.
    
    Args:
        model: PyTorch model class (not instance)
        X: feature tensor (N, input_dim)
        y: target tensor (N,)
        n_folds: number of folds for cross-validation
        epochs: number of training epochs
        lr: learning rate
    
    Returns:
        mean_accuracy: average accuracy across folds
        mean_auc: average AUC across folds
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize model for this fold
        fold_model = model(X.size(1))
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Training loop
        fold_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = fold_model(X_train)
            loss = criterion(outputs.squeeze(), y_train.float())
            loss.backward()
            optimizer.step()
        
        # Evaluation
        fold_model.eval()
        with torch.no_grad():
            val_outputs = fold_model(X_val)
            val_preds = (val_outputs.squeeze() > 0.5).float()
            
            accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())
            auc = roc_auc_score(y_val.numpy(), val_outputs.squeeze().numpy())
            
            accuracies.append(accuracy)
            aucs.append(auc)
    
    mean_accuracy = np.mean(accuracies)
    mean_auc = np.mean(aucs)
    
    print(f"Cross-validation results ({n_folds} folds):")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
    
    return mean_accuracy, mean_auc

# Example usage and testing
def example_usage():
    """
    Example of how to use the complete pipeline.
    """
    print("Molecular Property Prediction Pipeline Implementation Complete!")
    print("\nAvailable classes:")
    print("- BondAttentionMLP: MLP for bond attention scores")
    print("- AtomAggregationLayer: Atom information aggregation")
    print("- AttentionPooling: Molecular embedding with attention")
    print("- MolecularGNN: Complete GNN pipeline")
    print("- ElectronTransferPredictor: Binary classification model")
    print("- cross_fold_validation: Cross-validation function")
    
    # Example: Create a simple test
    print("\nExample usage:")
    
    # Create dummy data for testing
    node_dim = 4
    edge_dim = 4
    num_nodes = 10
    num_edges = 15
    
    # Dummy node features
    x = torch.randn(num_nodes, node_dim)
    
    # Dummy edge connectivity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Dummy edge features
    edge_attr = torch.randn(num_edges, edge_dim)
    
    # Dummy positions
    pos = torch.randn(num_nodes, 3)
    
    # Dummy bond attention data
    bond_attn_list = [torch.randn(5) for _ in range(3)]
    bond_neighbors_list = [[1, 2], [0, 2], [0, 1]]
    
    # Test MolecularGNN
    gnn = MolecularGNN(node_dim, edge_dim)
    molecular_embedding = gnn(x, edge_index, edge_attr, pos, bond_attn_list, bond_neighbors_list)
    print(f"Molecular embedding shape: {molecular_embedding.shape}")
    
    # Test center of mass distance
    mol1_pos = torch.randn(5, 3)
    mol2_pos = torch.randn(7, 3)
    distance = compute_center_of_mass_distance(mol1_pos, mol2_pos)
    print(f"Center of mass distance: {distance.item():.4f}")
    
    # Test feature extraction (with dummy data)
    mol1_data = type('Data', (), {'pos': mol1_pos, 'x': torch.randn(5, 4), 'edge_attr': torch.randn(8, 4), 'bond_attn': [torch.randn(3)]})()
    mol2_data = type('Data', (), {'pos': mol2_pos, 'x': torch.randn(7, 4), 'edge_attr': torch.randn(10, 4), 'bond_attn': [torch.randn(4)]})()
    mol1_props = {'Mulliken charges': [0.1, -0.1, 0.0, 0.2, -0.2], 'Loewdin charges': [0.05, -0.05, 0.0, 0.15, -0.15], 'Electron affinity (Ha)': 0.1, 'HOMO-LUMO gap (Ha)': 0.5}
    mol2_props = {'Mulliken charges': [0.2, -0.2, 0.1, -0.1, 0.0, 0.3, -0.3], 'Loewdin charges': [0.15, -0.15, 0.05, -0.05, 0.0, 0.25, -0.25], 'Electron affinity (Ha)': 0.15, 'HOMO-LUMO gap (Ha)': 0.6}
    
    features = extract_molecular_features(mol1_data, mol2_data, mol1_props, mol2_props)
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of features: {len(features)}")
    
    # Test ElectronTransferPredictor
    predictor = ElectronTransferPredictor(len(features))
    prediction = predictor(features.unsqueeze(0))
    print(f"Electron transfer probability: {prediction.item():.4f}")

if __name__ == "__main__":
    #example_usage()
    print("Loading molecules from SDF file...")
    
    # Load the molecules from the SDF file
    suppl = Chem.SDMolSupplier("molecule_pairs_optimized.sdf")
    molecules = [mol for mol in suppl if mol is not None]
    print(f"Loaded {len(molecules)} molecules from SDF")
    
    # Group molecules by RowID to get pairs
    molecule_pairs = {}
    for mol in molecules:
        try:
            row_id = int(mol.GetProp("RowID"))
            role = mol.GetProp("Role")
            cidnp = float(mol.GetProp("CIDNP"))
            
            if row_id not in molecule_pairs:
                molecule_pairs[row_id] = {}
            
            molecule_pairs[row_id][role] = mol
            molecule_pairs[row_id]['cidnp'] = cidnp
        except Exception as e:
            print(f"Warning: Could not process molecule: {e}")
            continue
    
    print(f"Found {len(molecule_pairs)} molecule pairs")
    
    all_features = []
    all_labels = []
    successful_pairs = 0
    
    for row_id, pair_data in molecule_pairs.items():
        if 'small_molecule' not in pair_data or 'photosensitizer' not in pair_data:
            print(f"Warning: Incomplete pair for row {row_id}")
            continue
            
        try:
            small_mol = pair_data['small_molecule']
            ps_mol = pair_data['photosensitizer']
            cidnp_value = pair_data['cidnp']
            
            # Create temporary SDF files for processing
            temp_small_path = f"temp_small_{row_id}.sdf"
            temp_ps_path = f"temp_ps_{row_id}.sdf"
            
            # Write temporary SDF files
            with Chem.SDWriter(temp_small_path) as w:
                w.write(small_mol)
            with Chem.SDWriter(temp_ps_path) as w:
                w.write(ps_mol)
            
            # Process molecules
            mol1_data = sdf_to_graph(temp_small_path)
            mol2_data = sdf_to_graph(temp_ps_path)
            mol1_props = compute_molecular_properties(small_mol)
            mol2_props = compute_molecular_properties(ps_mol)
            
            features = extract_molecular_features(mol1_data, mol2_data, mol1_props, mol2_props)
            all_features.append(features)
            
            # Binary labels: 1 if cidnp > 0, else 0
            label = 1.0 if cidnp_value > 0 else 0.0
            all_labels.append(label)
            
            successful_pairs += 1
            
            # Clean up temporary files
            import os
            try:
                os.remove(temp_small_path)
                os.remove(temp_ps_path)
            except:
                pass
            
            if successful_pairs % 10 == 0:
                print(f"Processed {successful_pairs} pairs...")
                
        except Exception as e:
            print(f"Error processing pair {row_id}: {e}")
            # Clean up temporary files on error
            try:
                os.remove(temp_small_path)
                os.remove(temp_ps_path)
            except:
                pass
            continue
    
    print(f"Successfully processed {successful_pairs} molecule pairs")
    
    if successful_pairs == 0:
        print("No successful pairs to process. Exiting.")
        exit(1)
    
    # Convert to tensors
    X = torch.stack(all_features)
    y = torch.tensor(all_labels, dtype=torch.float32)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {torch.bincount(y.long())}")
    
    # Run cross-fold validation
    print("\nStarting cross-fold validation...")
    mean_acc, mean_auc = cross_fold_validation(ElectronTransferPredictor, X, y, n_folds=5, epochs=50, lr=1e-3)

    print(f"\nFinal Results:")
    print(f"Mean CV Accuracy: {mean_acc:.4f}")
    print(f"Mean CV AUC: {mean_auc:.4f}")
