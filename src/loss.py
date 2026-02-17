import torch
import torch.nn.functional as F

class AnomalyLoss(torch.nn.Module):
    def __init__(self, lambda_param=1.0):
        super(AnomalyLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, embeddings, labels, center, mask):
        """
        embeddings: (N, dim)
        labels: (N,) 0 for Normal, 1 for Abnormal
        center: (dim,)
        mask: boolean mask for training nodes (Transductive setting)
        """
        # Filter by training mask
        h_train = embeddings[mask]
        y_train = labels[mask]
        
        normal_mask = (y_train == 0)
        abnormal_mask = (y_train == 1)
        
        h_norm = h_train[normal_mask]
        h_abnorm = h_train[abnormal_mask]
        
        # --- L_nor (Eq 5): Compactness of normal nodes ---
        if h_norm.size(0) > 0:
            dist_norm = torch.sum((h_norm - center) ** 2, dim=1)
            l_nor = torch.mean(dist_norm)
        else:
            l_nor = torch.tensor(0.0, device=embeddings.device)
            
        # --- L_AUC (Eq 6): Differential Approximation of AUC ---
        
        if h_norm.size(0) > 0 and h_abnorm.size(0) > 0:
            # Calculate anomaly scores a(u)
            score_norm = torch.sum((h_norm - center) ** 2, dim=1) # (N_norm,)
            score_abnorm = torch.sum((h_abnorm - center) ** 2, dim=1) # (N_ab,)
            
            # Pairwise differences: score_norm[i] - score_abnorm[j]
            # We want score_abnorm > score_norm, so this difference should be negative (large).
            # Sigmoid(negative) -> 0. 
            # If score_norm > score_abnorm, Sigmoid -> 1 (Penalty).
            
            diff_matrix = score_norm.unsqueeze(1) - score_abnorm.unsqueeze(0) # (N_norm, N_ab)
            l_auc = torch.mean(torch.sigmoid(diff_matrix))
        else:
            l_auc = torch.tensor(0.0, device=embeddings.device)
            
        # Total Loss (Eq 4)
        loss = l_nor - self.lambda_param * r_auc 
        
        return loss
