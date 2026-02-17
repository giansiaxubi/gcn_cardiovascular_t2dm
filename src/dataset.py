import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

class PopulationGraphBuilder:
    def __init__(self, 
                 df: pd.DataFrame, 
                 cont_cols: list, 
                 cat_cols: list, 
                 target_col: str, 
                 gamma: float = 0.6):
        """
        Args:
            df: Input dataframe.
            cont_cols: List of continuous column names.
            cat_cols: List of categorical column names.
            target_col: Name of the target column (0=Normal, 1=CVD).
            gamma: Threshold for adjacency matrix pruning.
        """
        self.df = df
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.gamma = gamma
        
        # Normalize continuous features s.t. L1 norm = 1 
        self.df[cont_cols] = self.df[cont_cols].div(self.df[cont_cols].abs().sum(axis=1), axis=0)
        
    def _compute_distance_matrix(self):
        """Implements Equation (1) from the paper."""
        N = len(self.df)
        cont_data = self.df[self.cont_cols].values
        cat_data = self.df[self.cat_cols].values
        
        # 1. Continuous Distance: ||xi - xj||^2
        # Using broadcasting: (N, 1, D) - (1, N, D)
        diff = cont_data[:, np.newaxis, :] - cont_data[np.newaxis, :, :]
        sq_dist_cont = np.sum(diff ** 2, axis=2) # (N, N)
        
        # Sigma is mean value of squared distances 
        sigma = np.mean(sq_dist_cont)
        
        # 2. Categorical Similarity: Sum of delta functions (matching features)
        # 1 if match, 0 if not.
        cat_diff = (cat_data[:, np.newaxis, :] == cat_data[np.newaxis, :, :]).astype(float)
        sum_delta_cat = np.sum(cat_diff, axis=2) # (N, N)
        S1_len = len(self.cat_cols)
        
        # Combine parts: exp(...) * (1/|S1|) * sum(delta)
        term1 = np.exp(-sq_dist_cont / (2 * (sigma**2)))   
        D_ij = term1 * (1 / (S1_len + 1e-8)) * sum_delta_cat
        
        # Set diagonal to 1
        np.fill_diagonal(D_ij, 1.0)
        
        self.adj_matrix_dense = D_ij # Store this as instance variable
        return D_ij

    def build_graph(self):
        # 1. Compute Adjacency Matrix A based on D(i,j)
        A_dense = self._compute_distance_matrix()
        
        # 2. Thresholding (Parameter Gamma)
        A_dense[A_dense < self.gamma] = 0
        A_dense[A_dense >= self.gamma] = 1
        
        # Convert to PyG Edge Index
        edge_indices = np.where(A_dense == 1)
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        
        # 3. Node Features (Concatenate Continuous + One-Hot Categorical)
        X_cont = torch.tensor(self.df[self.cont_cols].values, dtype=torch.float)
        
        # Simple One-Hot for categorical
        cat_dummies = pd.get_dummies(self.df[self.cat_cols], columns=self.cat_cols)
        X_cat = torch.tensor(cat_dummies.values, dtype=torch.float)
        
        X = torch.cat([X_cont, X_cat], dim=1)
        y = torch.tensor(self.df[self.target_col].values, dtype=torch.float)
        
        data = Data(x=X, edge_index=edge_index, y=y)
        data.adj_matrix = A_dense # Keep reference for debugging
        return data
