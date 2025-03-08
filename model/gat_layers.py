"""Modified from https://github.com/alexmonti19/dagnet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Enhanced Graph Attention Layer for SGTN, incorporating multi-head attention
    and improved spatial relationship handling.
    """
    def __init__(self, in_features, out_features, alpha, num_heads):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_heads = num_heads
        
        # Multi-head linear transformations
        self.W = nn.Parameter(
            torch.zeros(size=(num_heads, in_features, out_features))
        )
        self.a = nn.Parameter(
            torch.zeros(size=(num_heads, 2 * out_features, 1))
        )
        
        # Distance embedding layer
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, input, adj):
        batch_size = input.size(0)
        N = input.size(0)  # Number of nodes
        
        # Compute attention for each head
        multi_head_output = []
        for head in range(self.num_heads):
            # Transform input features
            h = torch.mm(input, self.W[head])  # N x out_features
            
            # Prepare attention inputs
            a_input = torch.cat([
                h.repeat(1, N).view(N * N, -1),
                h.repeat(N, 1)
            ], dim=1).view(N, -1, 2 * self.out_features)
            
            # Compute attention scores
            e = self.leakyrelu(torch.matmul(a_input, self.a[head]).squeeze(2))
            
            # Apply distance-based attention scaling
            if adj is not None:
                distances = -torch.log(adj + 1e-10)  # Convert adjacency to distances
                distance_weights = self.distance_embedding(distances.unsqueeze(-1))
                e = e + distance_weights.squeeze(-1)
            
            # Normalize attention scores
            attention = F.softmax(e, dim=1)
            
            # Apply attention
            head_output = torch.matmul(attention, h)
            multi_head_output.append(head_output)
        
        # Concatenate multi-head outputs
        output = torch.cat(multi_head_output, dim=1)
        
        # Average the concatenated outputs
        output = output.view(N, self.num_heads, -1).mean(dim=1)
        
        return F.elu(output)