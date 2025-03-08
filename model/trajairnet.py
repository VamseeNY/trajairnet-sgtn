import torch
from torch import nn
import torch.nn.functional as F
import math
from model.tcn_model import TemporalConvNet
from model.gat_model import GAT
from model.gat_layers import GraphAttentionLayer
from model.cvae_base import CVAE
from model.utils import acc_to_abs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SGTN(nn.Module):
    def __init__(self, args):
        super(SGTN, self).__init__()
        
        # Model dimensions
        self.input_size = args.input_channels
        self.n_classes = int(args.preds/args.preds_step)
        self.d_model = args.transformer_dim  # Transformer dimension
        self.nhead = args.transformer_heads  # Number of attention heads
        self.num_encoder_layers = args.transformer_layers
        self.dim_feedforward = args.transformer_ff_dim
        self.dropout = args.dropout
        
        # Spatial encoding (GNN) parameters
        self.graph_hidden = args.graph_hidden
        self.gat_heads = args.gat_heads
        self.alpha = args.alpha
        
        # Feature embedding layers
        self.input_embedding = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
        # Spatial Graph Neural Network
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                self.d_model if i == 0 else self.graph_hidden,
                self.graph_hidden,
                self.alpha,
                self.gat_heads
            ) for i in range(args.gnn_layers)
        ])
        
        # Context processing
        self.context_conv = nn.Conv1d(
            in_channels=args.num_context_input_c,
            out_channels=1,
            kernel_size=args.cnn_kernels
        )
        self.context_linear = nn.Linear(args.obs-1, args.num_context_output_c)
        
        # Trajectory prediction layers
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(self.graph_hidden + args.num_context_output_c, args.mlp_layer),
            nn.ReLU(),
            nn.Linear(args.mlp_layer, self.n_classes * self.input_size)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.graph_hidden + args.num_context_output_c, args.mlp_layer),
            nn.ReLU(),
            nn.Linear(args.mlp_layer, self.n_classes * self.input_size * 2)  # Mean and variance
        )
        
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, y, adj, context):
        batch_size = x.size(2)
        
        # Process each agent's trajectory
        encoded_trajectories = []
        encoded_contexts = []
        
        for agent in range(batch_size):
            # Extract and embed individual trajectory
            traj = torch.transpose(x[:, :, agent][None, :, :], 1, 2)  # Shape: [1, T, F]
            traj = traj.squeeze(0)  # Shape: [T, F]
            traj_embedded = self.input_embedding(traj)  # Shape: [T, D]
            traj_embedded = traj_embedded.unsqueeze(0)  # Shape: [1, T, D]
            traj_embedded = self.pos_encoder(traj_embedded)
            
            # Temporal encoding with Transformer
            temporal_encoding = self.transformer_encoder(traj_embedded)
            temporal_encoding = temporal_encoding.mean(dim=1)  # Average over time dimension
            
            # Process context information
            c = torch.transpose(context[:, :, agent][None, :, :], 1, 2)
            context_features = self.context_conv(c)
            context_features = F.relu(self.context_linear(context_features))
            
            encoded_trajectories.append(temporal_encoding)
            encoded_contexts.append(context_features)
            
        # Stack encoded features
        encoded_trajectories = torch.cat(encoded_trajectories, dim=0)
        encoded_contexts = torch.cat(encoded_contexts, dim=0)
        
        # Spatial encoding with GNN
        spatial_features = encoded_trajectories
        for gnn_layer in self.gnn_layers:
            spatial_features = gnn_layer(spatial_features, adj)
            
        # Combine spatial and temporal features
        combined_features = torch.cat([spatial_features, encoded_contexts.squeeze(1)], dim=-1)
        
        # Predict trajectories and uncertainties
        predicted_trajectories = self.trajectory_predictor(combined_features)
        uncertainties = self.uncertainty_estimator(combined_features)
        
        # Reshape predictions
        predicted_trajectories = predicted_trajectories.view(-1, self.n_classes, self.input_size)
        means, log_vars = torch.chunk(uncertainties.view(-1, self.n_classes, self.input_size * 2), 2, dim=-1)
        
        return predicted_trajectories, means, log_vars
    
    def inference(self, x, z, adj, context):
        # Similar to forward pass but without uncertainty estimation
        batch_size = x.size(2)
        
        encoded_trajectories = []
        encoded_contexts = []
        
        for agent in range(batch_size):
            traj = torch.transpose(x[:, :, agent][None, :, :], 1, 2)
            traj = traj.squeeze(0)
            traj_embedded = self.input_embedding(traj)
            traj_embedded = traj_embedded.unsqueeze(0)
            traj_embedded = self.pos_encoder(traj_embedded)
            temporal_encoding = self.transformer_encoder(traj_embedded)
            temporal_encoding = temporal_encoding.mean(dim=1)
            
            c = torch.transpose(context[:, :, agent][None, :, :], 1, 2)
            context_features = self.context_conv(c)
            context_features = F.relu(self.context_linear(context_features))
            
            encoded_trajectories.append(temporal_encoding)
            encoded_contexts.append(context_features)
            
        encoded_trajectories = torch.cat(encoded_trajectories, dim=0)
        encoded_contexts = torch.cat(encoded_contexts, dim=0)
        
        spatial_features = encoded_trajectories
        for gnn_layer in self.gnn_layers:
            spatial_features = gnn_layer(spatial_features, adj)
            
        combined_features = torch.cat([spatial_features, encoded_contexts.squeeze(1)], dim=-1)
        predicted_trajectories = self.trajectory_predictor(combined_features)
        predicted_trajectories = predicted_trajectories.view(-1, self.n_classes, self.input_size)
        
        return predicted_trajectories