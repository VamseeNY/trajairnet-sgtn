import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.trajairnet import SGTN
from model.utils import ade, fde, TrajectoryDataset, seq_collate

def test(model, loader_test, device):
    total_ade_loss = 0
    total_fde_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(loader_test):
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
            
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            
            # Create adjacency matrix based on proximity
            positions = obs_traj[-1, :, :2]  # Last observed positions
            distances = torch.cdist(positions, positions)
            adj = torch.exp(-distances / 10.0)  # Soft adjacency based on distance
            
            # Forward pass
            predicted_trajectories, _, _ = model(
                torch.transpose(obs_traj, 1, 2),
                pred_traj,
                adj,
                torch.transpose(context, 1, 2)
            )
            
            # Calculate metrics for each agent
            for i in range(num_agents):
                pred = predicted_trajectories[i].cpu().numpy()
                target = torch.transpose(pred_traj[:, :, i], 0, 1).cpu().numpy()
                
                total_ade_loss += ade(pred, target)
                total_fde_loss += fde(pred, target)
                total_samples += 1
    
    avg_ade = total_ade_loss / total_samples
    avg_fde = total_fde_loss / total_samples
    
    return avg_ade, avg_fde

def main():
    parser = argparse.ArgumentParser(description='Test SGTN model')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--epoch', type=int, required=True)
    
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    
    # Network params
    parser.add_argument('--input_channels', type=int, default=3)
    
    # Transformer parameters
    parser.add_argument('--transformer_dim', type=int, default=256)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=3)
    parser.add_argument('--transformer_ff_dim', type=int, default=1024)
    
    # GNN parameters
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--graph_hidden', type=int, default=256)
    parser.add_argument('--gat_heads', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.2)
    
    # Context and MLP parameters
    parser.add_argument('--num_context_input_c', type=int, default=2)
    parser.add_argument('--num_context_output_c', type=int, default=7)
    parser.add_argument('--cnn_kernels', type=int, default=2)
    parser.add_argument('--mlp_layer', type=int, default=256)
    
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--model_dir', type=str, default="/saved_models/")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print("Loading Test Data from", datapath + "test")
    dataset_test = TrajectoryDataset(
        datapath + "test",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        collate_fn=seq_collate
    )

    # Initialize and load model
    model = SGTN(args)
    model.to(device)

    model_path = os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_ade_loss, test_fde_loss = test(model, loader_test, device)
    print(f"Test ADE Loss: {test_ade_loss:.4f}, Test FDE Loss: {test_fde_loss:.4f}")

if __name__ == '__main__':
    main()

