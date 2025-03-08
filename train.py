import argparse
import os 
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from torch import optim

from model.trajairnet import SGTN
from model.utils import TrajectoryDataset, seq_collate, rmse
from test import test

def train():
    ##Dataset params
    parser=argparse.ArgumentParser(description='Train SGTN model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)

    ##Network params
    parser.add_argument('--input_channels',type=int,default=3)
    
    # Transformer parameters
    parser.add_argument('--transformer_dim',type=int,default=256)
    parser.add_argument('--transformer_heads',type=int,default=8)
    parser.add_argument('--transformer_layers',type=int,default=3)
    parser.add_argument('--transformer_ff_dim',type=int,default=1024)
    
    # GNN parameters
    parser.add_argument('--gnn_layers',type=int,default=3)
    parser.add_argument('--graph_hidden',type=int,default=256)
    parser.add_argument('--gat_heads',type=int,default=8)
    parser.add_argument('--alpha',type=float,default=0.2)
    
    # Context and MLP parameters
    parser.add_argument('--num_context_input_c',type=int,default=2)
    parser.add_argument('--num_context_output_c',type=int,default=7)
    parser.add_argument('--cnn_kernels',type=int,default=2)
    parser.add_argument('--mlp_layer',type=int,default=256)
    
    # Training parameters
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    parser.add_argument('--total_epochs',type=int,default=50)
    parser.add_argument('--warmup_steps',type=int,default=4000)
    
    parser.add_argument('--delim',type=str,default=' ')
    parser.add_argument('--evaluate',type=bool,default=True)
    parser.add_argument('--save_model',type=bool,default=True)
    parser.add_argument('--model_pth',type=str,default="/saved_models/")

    args=parser.parse_args()

    ##Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load test and train data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from ",datapath + "train")
    dataset_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    loader_train = DataLoader(dataset_train,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)

    model = SGTN(args)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    num_batches = len(loader_train)
 
    print("Starting Training....")

    for epoch in range(1, args.total_epochs+1):
        model.train()
        total_loss = 0
        total_batches = 0

        for batch in tqdm(loader_train):
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
            
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            
            # Create adjacency matrix based on proximity
            positions = obs_traj[-1, :, :2]  # Last observed positions
            distances = torch.cdist(positions, positions)
            adj = torch.exp(-distances / 10.0)  # Soft adjacency based on distance
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_trajectories, means, log_vars = model(
                torch.transpose(obs_traj, 1, 2),
                pred_traj,
                adj,
                torch.transpose(context, 1, 2)
            )
            
            # Compute losses
            reconstruction_loss = rmse(predicted_trajectories, torch.transpose(pred_traj, 0, 1))
            kld_loss = -0.5 * torch.mean(1 + log_vars - means.pow(2) - log_vars.exp())
            
            # Total loss with KL divergence weight annealing
            kld_weight = min(epoch / 10, 1.0)  # Gradually increase KLD weight
            loss = reconstruction_loss + kld_weight * kld_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        scheduler.step()
        
        print(f"EPOCH: {epoch}, Train Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if args.save_model:
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at", model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, model_path)
        
        if args.evaluate:
            print("Starting Testing....")
            model.eval()
            test_ade_loss, test_fde_loss = test(model, loader_test, device)
            print(f"EPOCH: {epoch}, Train Loss: {avg_loss:.4f}, Test ADE Loss: {test_ade_loss:.4f}, Test FDE Loss: {test_fde_loss:.4f}")

if __name__ == "__main__":
    train()