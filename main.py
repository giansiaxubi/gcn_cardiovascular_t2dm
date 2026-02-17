import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from src.dataset import PopulationGraphBuilder
from src.model import GCNAnomaly
from src.loss import AnomalyLoss
from src.utils import probability_transform, evaluate_metrics

def main(args):
    # 1. Load Data
    df = pd.read_csv(args.data_path)
    
    # Define columns (User must adapt this list to their specific CSV)
    # Example placeholders based on paper description
    target_col = args.target_col
    # Assume all other cols are features for this example
    feature_cols = [c for c in df.columns if c != target_col]
    # Simple heuristic to split cat/cont (User should specify manually in prod)
    cat_cols = [c for c in feature_cols if df[c].nunique() < 10]
    cont_cols = [c for c in feature_cols if c not in cat_cols]
    
    print(f"Building Graph with {len(df)} patients...")
    builder = PopulationGraphBuilder(df, cont_cols, cat_cols, target_col, gamma=args.gamma)
    data = builder.build_graph()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # 2. Nested Cross Validation (Simplified to 1 Loop for demo)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_numpy = data.y.cpu().numpy()
    
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_numpy)), y_numpy)):
        print(f"\nFold {fold+1}...")
        
        # Create masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        # Initialize Model
        model = GCNAnomaly(in_channels=data.num_features,
                           hidden_channels=args.hidden_dim,
                           out_channels=args.output_dim,
                           dropout=args.dropout).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = AnomalyLoss(lambda_param=args.lambda_loss)
        
        # --- Center 'c' Initialization (Crucial Step) ---
        # "Calculated as the mean of embeddings for the normal-class nodes... 
        # in the initial forward pass with randomly initialized weights"
        model.eval()
        with torch.no_grad():
            initial_embeds = model(data.x, data.edge_index)
            # Filter normal nodes IN TRAINING SET
            normal_train_indices = train_mask & (data.y == 0)
            if normal_train_indices.sum() > 0:
                center = initial_embeds[normal_train_indices].mean(dim=0)
            else:
                center = torch.zeros(args.output_dim).to(device)
            
            model.center = center.detach() # Fix center
            
        # Training
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            loss = criterion(embeddings, data.y, model.center, train_mask)
            loss.backward()
            optimizer.step()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            dists, _ = model.get_anomaly_score(data.x, data.edge_index)
            
            # Calculate Mu based on training set mean score
            train_scores = dists[train_mask].cpu().numpy()
            mu = 1.0 / (np.mean(train_scores) + 1e-8)
            
            # Predict on Validation
            val_scores = dists[val_mask].cpu().numpy()
            val_probs = probability_transform(val_scores, mu)
            val_true = data.y[val_mask].cpu().numpy()
            
            fold_metrics = evaluate_metrics(val_true, val_probs)
            print(f"Fold {fold+1} metrics: {fold_metrics}")
            results.append(fold_metrics)

    # Aggregate Results
    avg_auc = np.mean([r['AUC'] for r in results])
    print(f"\nAverage AUC across folds: {avg_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/dataset.csv')
    parser.add_argument('--target_col', type=str, default='CVD_Outcome')
    parser.add_argument('--gamma', type=float, default=0.6)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--output_dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lambda_loss', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    main(args)
