import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model.trajairnet import SGTN
from model.utils import TrajectoryDataset, seq_collate

def plot_trajectory(obs_traj, pred_traj, predicted_traj, means, log_vars, save_path):
    """Plot trajectories with uncertainty bounds"""
    plt.figure(figsize=(12, 8))
    
    # Plot observed trajectory
    obs_x = obs_traj[:, 0]
    obs_y = obs_traj[:, 1]
    plt.plot(obs_x, obs_y, 'b-', label='Observed', linewidth=2)
    plt.plot(obs_x[-1], obs_y[-1], 'b*', markersize=10, label='Last Observed')
    
    # Plot ground truth trajectory
    pred_x = pred_traj[:, 0]
    pred_y = pred_traj[:, 1]
    plt.plot(pred_x, pred_y, 'g--', label='Ground Truth', linewidth=2)
    
    # Plot predicted trajectory with uncertainty
    pred_x = predicted_traj[:, 0]
    pred_y = predicted_traj[:, 1]
    std_x = np.exp(0.5 * log_vars[:, 0])
    std_y = np.exp(0.5 * log_vars[:, 1])
    
    plt.plot(pred_x, pred_y, 'r-', label='Prediction', linewidth=2)
    plt.fill_between(pred_x, pred_y - 2*std_y, pred_y + 2*std_y, 
                     color='red', alpha=0.2, label='95% Confidence')
    
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title('Aircraft Trajectory Prediction', fontsize=14)
    
    # Add scale bar
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(args):
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
        shuffle=False,  # Don't shuffle for visualization
        collate_fn=seq_collate
    )
    
    # Load model
    model = SGTN(args)
    model.to(device)
    
    model_path = os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + str(args.epoch) + ".pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'visualization_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader_test)):
            if batch_idx >= args.num_viz:  # Only visualize specified number of samples
                break
                
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
            
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            
            # Create adjacency matrix based on proximity
            positions = obs_traj[-1, :, :2]
            distances = torch.cdist(positions, positions)
            adj = torch.exp(-distances / 10.0)
            
            # Forward pass
            predicted_trajectories, means, log_vars = model(
                torch.transpose(obs_traj, 1, 2),
                pred_traj,
                adj,
                torch.transpose(context, 1, 2)
            )
            
            # Plot each agent's trajectory
            for agent in range(num_agents):
                obs = obs_traj[:, agent, :].cpu().numpy()
                pred = torch.transpose(pred_traj[:, :, agent], 0, 1).cpu().numpy()
                pred_traj = predicted_trajectories[agent].cpu().numpy()
                agent_means = means[agent].cpu().numpy()
                agent_log_vars = log_vars[agent].cpu().numpy()
                
                save_path = os.path.join(output_dir, f'trajectory_batch{batch_idx}_agent{agent}.png')
                plot_trajectory(obs, pred, pred_traj, agent_means, agent_log_vars, save_path)
    
    print(f"Visualizations saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize SGTN predictions')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--num_viz', type=int, default=10, help='Number of samples to visualize')
    
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
    visualize_predictions(args)

if __name__ == '__main__':
    main() 