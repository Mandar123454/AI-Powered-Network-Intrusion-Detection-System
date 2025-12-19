"""
GNN-Based Network Intrusion Detection
=====================================
Graph Neural Network detector for network topology-aware intrusion detection.
Treats the network as a dynamic graph: nodes = devices, edges = connections.

This represents state-of-the-art network security AI:
- Captures network topology and device relationships
- Detects lateral movement and multi-stage attacks
- Identifies anomalous communication patterns
- Learns normal graph structure for deviation detection

Architecture:
- GraphSAGE layers for scalable neighborhood aggregation
- Graph Attention Networks (GAT) for weighted neighbor importance
- Temporal graph convolutions for time-aware detection
- Hierarchical pooling for multi-scale pattern detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, GATConv, GCNConv, GraphConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class NetworkNode:
    """Represents a device/host in the network graph."""
    node_id: str
    ip_address: str
    mac_address: Optional[str] = None
    hostname: Optional[str] = None
    device_type: str = "unknown"  # server, workstation, iot, router, etc.
    os_fingerprint: Optional[str] = None
    
    # Behavioral features
    avg_bytes_sent: float = 0.0
    avg_bytes_recv: float = 0.0
    avg_connections: float = 0.0
    unique_ports_used: int = 0
    protocol_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Temporal features
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    activity_hours: List[int] = field(default_factory=list)
    
    # Risk indicators
    risk_score: float = 0.0
    is_known_malicious: bool = False
    threat_intel_hits: int = 0


@dataclass
class NetworkEdge:
    """Represents a connection/flow between two nodes."""
    source_id: str
    target_id: str
    
    # Connection features
    total_bytes: int = 0
    total_packets: int = 0
    connection_count: int = 0
    
    # Protocol/port info
    protocols: Dict[str, int] = field(default_factory=dict)
    destination_ports: Dict[int, int] = field(default_factory=dict)
    
    # Temporal features
    first_connection: Optional[datetime] = None
    last_connection: Optional[datetime] = None
    avg_duration: float = 0.0
    
    # Anomaly indicators
    has_suspicious_ports: bool = False
    encrypted_ratio: float = 0.0
    bidirectional: bool = True


class GraphAttentionLayer(nn.Module):
    """
    Custom Graph Attention Layer with multi-head attention.
    Computes attention weights for each neighbor.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformations for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout_layer = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention computation.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge indices [2, E]
            return_attention: Whether to return attention weights
            
        Returns:
            Updated node features and optional attention weights
        """
        num_nodes = x.size(0)
        
        # Transform features for each head: [N, heads, out_features]
        h = torch.einsum('ni,hio->nho', x, self.W)
        
        # Compute attention scores
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Source and destination attention components
        e_src = torch.einsum('nho,hok->nhk', h, self.a_src).squeeze(-1)  # [N, heads]
        e_dst = torch.einsum('nho,hok->nhk', h, self.a_dst).squeeze(-1)  # [N, heads]
        
        # Attention for each edge
        e = e_src[src_idx] + e_dst[dst_idx]  # [E, heads]
        e = self.leaky_relu(e)
        
        # Softmax over neighbors (using scatter operations for efficiency)
        alpha = self._sparse_softmax(e, dst_idx, num_nodes)
        alpha = self.dropout_layer(alpha)
        
        # Aggregate neighbor features with attention
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        h_src = h[src_idx]  # [E, heads, out_features]
        weighted_h = alpha.unsqueeze(-1) * h_src  # [E, heads, out_features]
        
        # Scatter add for aggregation
        out.scatter_add_(0, dst_idx.view(-1, 1, 1).expand_as(weighted_h), weighted_h)
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(num_nodes, -1)  # [N, heads * out_features]
        else:
            out = out.mean(dim=1)  # [N, out_features]
        
        if return_attention:
            return out, alpha
        return out, None
    
    def _sparse_softmax(
        self,
        e: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Compute softmax over edge attention for each destination node."""
        # Subtract max for numerical stability
        e_max = torch.zeros(num_nodes, e.size(1), device=e.device)
        e_max.scatter_reduce_(0, dst_idx.view(-1, 1).expand_as(e), e, reduce='amax')
        e = e - e_max[dst_idx]
        
        # Exp and sum
        exp_e = torch.exp(e)
        sum_exp = torch.zeros(num_nodes, e.size(1), device=e.device)
        sum_exp.scatter_add_(0, dst_idx.view(-1, 1).expand_as(exp_e), exp_e)
        
        return exp_e / (sum_exp[dst_idx] + 1e-10)


class TemporalGraphConv(nn.Module):
    """
    Temporal Graph Convolution Layer.
    Processes sequences of graph snapshots with temporal attention.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_time_steps: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_time_steps = num_time_steps
        
        # Spatial graph convolution
        self.spatial_conv = GATConv(
            in_features, hidden_features,
            heads=4, concat=False, dropout=dropout
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_features, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Temporal positional encoding
        self.time_embedding = nn.Embedding(num_time_steps, hidden_features)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_sequence: List[torch.Tensor],
        edge_index_sequence: List[torch.Tensor],
        node_mapping: Optional[Dict[str, int]] = None
    ) -> torch.Tensor:
        """
        Process temporal graph sequence.
        
        Args:
            x_sequence: List of node features for each time step
            edge_index_sequence: List of edge indices for each time step
            node_mapping: Mapping from node IDs to indices
            
        Returns:
            Temporal-aware node embeddings
        """
        # Get max nodes across all time steps
        max_nodes = max(x.size(0) for x in x_sequence)
        num_steps = len(x_sequence)
        device = x_sequence[0].device
        
        # Process each time step through spatial convolution
        temporal_embeddings = []
        for t, (x, edge_index) in enumerate(zip(x_sequence, edge_index_sequence)):
            # Spatial convolution
            h = self.spatial_conv(x, edge_index)
            
            # Pad to max_nodes
            if h.size(0) < max_nodes:
                padding = torch.zeros(max_nodes - h.size(0), h.size(1), device=device)
                h = torch.cat([h, padding], dim=0)
            
            # Add temporal position
            time_pos = self.time_embedding(torch.tensor([t], device=device))
            h = h + time_pos
            
            temporal_embeddings.append(h)
        
        # Stack temporal embeddings: [max_nodes, num_steps, hidden]
        temporal_stack = torch.stack(temporal_embeddings, dim=1)
        
        # Temporal attention across time steps
        attn_output, _ = self.temporal_attention(
            temporal_stack, temporal_stack, temporal_stack
        )
        
        # Take the last time step's representation
        out = attn_output[:, -1, :]
        out = self.output_proj(out)
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out


class HierarchicalGraphPooling(nn.Module):
    """
    Hierarchical graph pooling for multi-scale pattern detection.
    Pools nodes into clusters, then clusters into superclusters.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_clusters: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.num_clusters = num_clusters
        
        # Cluster assignment network
        self.cluster_assign = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_clusters),
            nn.Softmax(dim=-1)
        )
        
        # Cluster feature aggregation
        self.cluster_conv = GraphConv(in_features, hidden_features)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_features * 3, hidden_features)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical pooling of graph.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge indices [2, E]
            batch: Batch assignment for each node
            
        Returns:
            Graph-level embedding and cluster assignments
        """
        # Get soft cluster assignments
        cluster_probs = self.cluster_assign(x)  # [N, num_clusters]
        
        # Aggregate features into clusters
        cluster_features = torch.mm(cluster_probs.t(), x)  # [num_clusters, in_features]
        
        # Apply graph convolution on node level
        conv_features = self.cluster_conv(x, edge_index)  # [N, hidden]
        
        # Aggregate to cluster level
        cluster_conv = torch.mm(cluster_probs.t(), conv_features)  # [num_clusters, hidden]
        
        # Multiple pooling strategies
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        mean_pool = global_mean_pool(conv_features, batch)
        max_pool = global_max_pool(conv_features, batch)
        add_pool = global_add_pool(conv_features, batch)
        
        # Combine pooling strategies
        graph_embed = torch.cat([mean_pool, max_pool, add_pool], dim=-1)
        graph_embed = self.output_proj(graph_embed)
        
        return graph_embed, cluster_probs


class GNNIntrusionDetector(nn.Module):
    """
    Complete GNN-based Network Intrusion Detection System.
    
    Architecture:
    1. Node Feature Encoder: Encodes raw device features
    2. Edge Feature Encoder: Encodes connection features
    3. Multi-layer Graph Attention: Captures topology patterns
    4. Temporal Graph Processing: Handles time-varying graphs
    5. Hierarchical Pooling: Multi-scale pattern detection
    6. Anomaly Classifier: Classifies attacks and anomalies
    """
    
    def __init__(
        self,
        node_features: int = 32,
        edge_features: int = 16,
        hidden_dim: int = 128,
        num_classes: int = 10,  # Normal + 9 attack types
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_temporal: bool = True,
        num_time_steps: int = 10
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_temporal = use_temporal
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge feature encoder (for edge-weighted graphs)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Multi-layer Graph Attention Network
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        for i in range(num_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    in_dim, hidden_dim,
                    heads=num_heads,
                    concat=True if i < num_gat_layers - 1 else False,
                    dropout=dropout
                )
            )
            out_dim = hidden_dim * num_heads if i < num_gat_layers - 1 else hidden_dim
            self.gat_norms.append(nn.LayerNorm(out_dim))
        
        # Temporal graph processing
        if use_temporal:
            self.temporal_conv = TemporalGraphConv(
                hidden_dim, hidden_dim, hidden_dim,
                num_time_steps=num_time_steps,
                dropout=dropout
            )
        
        # Hierarchical pooling
        self.hierarchical_pool = HierarchicalGraphPooling(
            hidden_dim, hidden_dim,
            num_clusters=10,
            dropout=dropout
        )
        
        # Node-level anomaly classifier
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Graph-level anomaly classifier
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Normal/Anomalous graph
        )
        
        # Edge anomaly detector
        self.edge_anomaly = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Attack pattern embeddings for few-shot learning
        self.attack_embeddings = nn.Embedding(num_classes, hidden_dim)
        
        self._init_attack_embeddings()
    
    def _init_attack_embeddings(self):
        """Initialize attack pattern embeddings with semantic meaning."""
        # Attack types: Normal, DoS, Probe, R2L, U2R, Botnet, Lateral, C2, Exfil, APT
        # Initialize with distinct patterns
        nn.init.orthogonal_(self.attack_embeddings.weight)
    
    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw node features."""
        return self.node_encoder(x)
    
    def encode_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode raw edge features."""
        return self.edge_encoder(edge_attr)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for intrusion detection.
        
        Args:
            x: Node features [N, node_features]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_features]
            batch: Batch assignment [N]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with node and graph predictions
        """
        # Encode node features
        h = self.encode_nodes(x)
        
        # Apply GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            h_new = gat(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            
            # Residual connection (with projection if needed)
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Node-level predictions
        node_logits = self.node_classifier(h)
        
        # Graph-level predictions
        graph_embed, cluster_assign = self.hierarchical_pool(h, edge_index, batch)
        graph_logits = self.graph_classifier(graph_embed)
        
        # Edge anomaly scores
        edge_anomaly_scores = None
        if edge_attr is not None:
            src, dst = edge_index[0], edge_index[1]
            edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            edge_anomaly_scores = self.edge_anomaly(edge_repr)
        
        result = {
            'node_logits': node_logits,
            'node_probs': F.softmax(node_logits, dim=-1),
            'graph_logits': graph_logits,
            'graph_probs': F.softmax(graph_logits, dim=-1),
            'edge_anomaly_scores': edge_anomaly_scores,
            'cluster_assignments': cluster_assign
        }
        
        if return_embeddings:
            result['node_embeddings'] = h
            result['graph_embedding'] = graph_embed
        
        return result
    
    def detect_lateral_movement(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        suspected_source: int,
        hop_limit: int = 3
    ) -> Dict[str, Any]:
        """
        Detect lateral movement from a suspected compromised node.
        
        Args:
            node_embeddings: Node embeddings from forward pass
            edge_index: Edge indices
            suspected_source: Index of suspected source node
            hop_limit: Maximum hops to analyze
            
        Returns:
            Lateral movement analysis
        """
        device = node_embeddings.device
        num_nodes = node_embeddings.size(0)
        
        # Build adjacency for BFS
        adj = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src].append(dst)
        
        # BFS to find reachable nodes
        visited = {suspected_source}
        current_hop = {suspected_source}
        hop_nodes = {0: [suspected_source]}
        
        for hop in range(1, hop_limit + 1):
            next_hop = set()
            for node in current_hop:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_hop.add(neighbor)
            
            if next_hop:
                hop_nodes[hop] = list(next_hop)
                current_hop = next_hop
            else:
                break
        
        # Compute similarity to source embedding
        source_embed = node_embeddings[suspected_source]
        similarities = F.cosine_similarity(
            node_embeddings,
            source_embed.unsqueeze(0),
            dim=-1
        )
        
        # Identify anomalous paths (high similarity = potential lateral movement)
        risk_scores = {}
        for hop, nodes in hop_nodes.items():
            for node in nodes:
                risk_scores[node] = {
                    'hop_distance': hop,
                    'embedding_similarity': similarities[node].item(),
                    'risk_score': similarities[node].item() * (0.9 ** hop)  # Decay by distance
                }
        
        # Sort by risk
        sorted_risks = sorted(
            risk_scores.items(),
            key=lambda x: x[1]['risk_score'],
            reverse=True
        )
        
        return {
            'source_node': suspected_source,
            'reachable_nodes': len(visited),
            'hop_distribution': {k: len(v) for k, v in hop_nodes.items()},
            'risk_ranked_nodes': sorted_risks[:20],  # Top 20 risky nodes
            'high_risk_count': sum(1 for _, r in sorted_risks if r['risk_score'] > 0.7)
        }
    
    def compute_anomaly_score(
        self,
        output: Dict[str, torch.Tensor],
        normal_class_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute overall anomaly score for each node.
        
        Args:
            output: Forward pass output
            normal_class_idx: Index of the normal class
            
        Returns:
            Anomaly scores [N]
        """
        node_probs = output['node_probs']
        
        # Anomaly score = 1 - P(normal)
        anomaly_scores = 1.0 - node_probs[:, normal_class_idx]
        
        return anomaly_scores


class NetworkGraphBuilder:
    """
    Builds dynamic network graphs from flow data.
    Converts raw network traffic into graph structures for GNN processing.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 32,
        edge_feature_dim: int = 16,
        max_nodes: int = 10000,
        time_window: timedelta = timedelta(minutes=5)
    ):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.max_nodes = max_nodes
        self.time_window = time_window
        
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[Tuple[str, str], NetworkEdge] = {}
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        
        # Feature statistics for normalization
        self.feature_stats = {
            'node_mean': None,
            'node_std': None,
            'edge_mean': None,
            'edge_std': None
        }
    
    def _get_node_id(self, ip: str) -> str:
        """Generate stable node ID from IP."""
        return hashlib.md5(ip.encode()).hexdigest()[:12]
    
    def add_flow(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        bytes_sent: int,
        bytes_recv: int,
        packets: int,
        duration: float,
        timestamp: datetime,
        flags: Optional[Dict[str, Any]] = None
    ):
        """Add a network flow to the graph."""
        # Update or create source node
        src_id = self._get_node_id(src_ip)
        if src_id not in self.nodes:
            self.nodes[src_id] = NetworkNode(
                node_id=src_id,
                ip_address=src_ip,
                first_seen=timestamp
            )
            idx = len(self.node_to_idx)
            self.node_to_idx[src_id] = idx
            self.idx_to_node[idx] = src_id
        
        src_node = self.nodes[src_id]
        src_node.last_seen = timestamp
        src_node.avg_bytes_sent = 0.9 * src_node.avg_bytes_sent + 0.1 * bytes_sent
        src_node.avg_connections = 0.9 * src_node.avg_connections + 0.1
        
        # Update or create destination node
        dst_id = self._get_node_id(dst_ip)
        if dst_id not in self.nodes:
            self.nodes[dst_id] = NetworkNode(
                node_id=dst_id,
                ip_address=dst_ip,
                first_seen=timestamp
            )
            idx = len(self.node_to_idx)
            self.node_to_idx[dst_id] = idx
            self.idx_to_node[idx] = dst_id
        
        dst_node = self.nodes[dst_id]
        dst_node.last_seen = timestamp
        dst_node.avg_bytes_recv = 0.9 * dst_node.avg_bytes_recv + 0.1 * bytes_recv
        
        # Update or create edge
        edge_key = (src_id, dst_id)
        if edge_key not in self.edges:
            self.edges[edge_key] = NetworkEdge(
                source_id=src_id,
                target_id=dst_id,
                first_connection=timestamp
            )
        
        edge = self.edges[edge_key]
        edge.total_bytes += bytes_sent + bytes_recv
        edge.total_packets += packets
        edge.connection_count += 1
        edge.last_connection = timestamp
        edge.avg_duration = 0.9 * edge.avg_duration + 0.1 * duration
        
        # Update protocol stats
        edge.protocols[protocol] = edge.protocols.get(protocol, 0) + 1
        edge.destination_ports[dst_port] = edge.destination_ports.get(dst_port, 0) + 1
        
        # Check for suspicious ports
        suspicious_ports = {22, 23, 3389, 445, 135, 139, 4444, 5555, 6666, 31337}
        if dst_port in suspicious_ports:
            edge.has_suspicious_ports = True
    
    def _extract_node_features(self, node: NetworkNode) -> np.ndarray:
        """Extract feature vector from node."""
        features = np.zeros(self.node_feature_dim)
        
        # Basic traffic features (0-5)
        features[0] = np.log1p(node.avg_bytes_sent)
        features[1] = np.log1p(node.avg_bytes_recv)
        features[2] = np.log1p(node.avg_connections)
        features[3] = node.unique_ports_used / 100.0
        features[4] = node.risk_score
        features[5] = 1.0 if node.is_known_malicious else 0.0
        
        # Device type encoding (6-12)
        device_types = ['server', 'workstation', 'iot', 'router', 'printer', 'mobile', 'unknown']
        if node.device_type in device_types:
            features[6 + device_types.index(node.device_type)] = 1.0
        
        # Temporal features (13-20)
        if node.first_seen and node.last_seen:
            age = (node.last_seen - node.first_seen).total_seconds()
            features[13] = np.log1p(age)
        
        # Activity pattern (hourly distribution)
        if node.activity_hours:
            hour_counts = np.zeros(24)
            for hour in node.activity_hours:
                hour_counts[hour] += 1
            hour_probs = hour_counts / (hour_counts.sum() + 1e-10)
            # Entropy of activity pattern
            features[14] = -np.sum(hour_probs * np.log(hour_probs + 1e-10))
        
        # Protocol distribution entropy (15-20)
        if node.protocol_distribution:
            probs = np.array(list(node.protocol_distribution.values()))
            probs = probs / (probs.sum() + 1e-10)
            features[15] = -np.sum(probs * np.log(probs + 1e-10))
        
        # Threat intelligence (20-25)
        features[20] = min(node.threat_intel_hits, 10) / 10.0
        
        return features
    
    def _extract_edge_features(self, edge: NetworkEdge) -> np.ndarray:
        """Extract feature vector from edge."""
        features = np.zeros(self.edge_feature_dim)
        
        # Traffic volume (0-3)
        features[0] = np.log1p(edge.total_bytes)
        features[1] = np.log1p(edge.total_packets)
        features[2] = np.log1p(edge.connection_count)
        features[3] = np.log1p(edge.avg_duration)
        
        # Protocol diversity (4-6)
        num_protocols = len(edge.protocols)
        features[4] = min(num_protocols, 10) / 10.0
        
        # Port diversity (5-7)
        num_ports = len(edge.destination_ports)
        features[5] = min(num_ports, 100) / 100.0
        
        # Suspicious indicators (6-10)
        features[6] = 1.0 if edge.has_suspicious_ports else 0.0
        features[7] = edge.encrypted_ratio
        features[8] = 1.0 if edge.bidirectional else 0.0
        
        # Temporal features (9-12)
        if edge.first_connection and edge.last_connection:
            duration = (edge.last_connection - edge.first_connection).total_seconds()
            features[9] = np.log1p(duration)
            
            # Connection frequency
            if duration > 0:
                features[10] = edge.connection_count / duration
        
        return features
    
    def build_graph(self, device: str = 'cpu') -> Data:
        """Build PyTorch Geometric graph from current state."""
        num_nodes = len(self.nodes)
        
        if num_nodes == 0:
            # Return empty graph
            return Data(
                x=torch.zeros(1, self.node_feature_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, self.edge_feature_dim)
            )
        
        # Build node features
        node_features = np.zeros((num_nodes, self.node_feature_dim))
        for node_id, idx in self.node_to_idx.items():
            node_features[idx] = self._extract_node_features(self.nodes[node_id])
        
        # Normalize node features
        if self.feature_stats['node_mean'] is None:
            self.feature_stats['node_mean'] = node_features.mean(axis=0)
            self.feature_stats['node_std'] = node_features.std(axis=0) + 1e-10
        
        node_features = (node_features - self.feature_stats['node_mean']) / self.feature_stats['node_std']
        
        # Build edge index and features
        edge_list = []
        edge_features = []
        
        for (src_id, dst_id), edge in self.edges.items():
            if src_id in self.node_to_idx and dst_id in self.node_to_idx:
                src_idx = self.node_to_idx[src_id]
                dst_idx = self.node_to_idx[dst_id]
                edge_list.append([src_idx, dst_idx])
                edge_features.append(self._extract_edge_features(edge))
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, self.edge_feature_dim)
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return data.to(device)
    
    def get_node_by_ip(self, ip: str) -> Optional[int]:
        """Get node index by IP address."""
        node_id = self._get_node_id(ip)
        return self.node_to_idx.get(node_id)
    
    def get_ip_by_node(self, idx: int) -> Optional[str]:
        """Get IP address by node index."""
        node_id = self.idx_to_node.get(idx)
        if node_id and node_id in self.nodes:
            return self.nodes[node_id].ip_address
        return None
    
    def prune_old_data(self, cutoff_time: datetime):
        """Remove nodes and edges older than cutoff time."""
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            if node.last_seen and node.last_seen < cutoff_time:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            if node_id in self.node_to_idx:
                idx = self.node_to_idx[node_id]
                del self.node_to_idx[node_id]
                del self.idx_to_node[idx]
        
        edges_to_remove = []
        for edge_key, edge in self.edges.items():
            if edge.last_connection and edge.last_connection < cutoff_time:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
        
        # Rebuild index mapping
        self.node_to_idx = {}
        self.idx_to_node = {}
        for idx, node_id in enumerate(self.nodes.keys()):
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'avg_degree': len(self.edges) * 2 / max(len(self.nodes), 1),
            'device_types': dict(
                (dt, sum(1 for n in self.nodes.values() if n.device_type == dt))
                for dt in set(n.device_type for n in self.nodes.values())
            ),
            'total_bytes': sum(e.total_bytes for e in self.edges.values()),
            'suspicious_edges': sum(1 for e in self.edges.values() if e.has_suspicious_ports)
        }


class GNNTrainer:
    """Training utilities for the GNN detector."""
    
    def __init__(
        self,
        model: GNNIntrusionDetector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Class weights for imbalanced data
        self.class_weights = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def set_class_weights(self, class_counts: Dict[int, int]):
        """Set class weights based on class distribution."""
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weights.append(total / (num_classes * count))
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def train_step(
        self,
        data: Data,
        labels: torch.Tensor,
        graph_label: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        output = self.model(
            data.x, data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            batch=data.batch if hasattr(data, 'batch') else None
        )
        
        # Node classification loss
        if self.class_weights is not None:
            node_loss = F.cross_entropy(output['node_logits'], labels, weight=self.class_weights)
        else:
            node_loss = F.cross_entropy(output['node_logits'], labels)
        
        # Graph classification loss (if provided)
        graph_loss = 0.0
        if graph_label is not None:
            graph_label = graph_label.to(self.device)
            graph_loss = F.cross_entropy(output['graph_logits'], graph_label)
        
        # Total loss
        total_loss = node_loss + 0.1 * graph_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy
        preds = output['node_logits'].argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
        
        return {
            'loss': total_loss.item(),
            'node_loss': node_loss.item(),
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        data: Data,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate on validation data."""
        self.model.eval()
        
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        output = self.model(
            data.x, data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            batch=data.batch if hasattr(data, 'batch') else None
        )
        
        loss = F.cross_entropy(output['node_logits'], labels)
        
        preds = output['node_logits'].argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
        
        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
        
        class_accuracy = {
            k: class_correct[k] / class_total[k]
            for k in class_total
        }
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'class_accuracy': class_accuracy
        }
    
    def fit(
        self,
        train_data: List[Tuple[Data, torch.Tensor]],
        val_data: List[Tuple[Data, torch.Tensor]],
        epochs: int = 100,
        early_stopping: int = 10
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            train_accs = []
            
            for data, labels in train_data:
                metrics = self.train_step(data, labels)
                train_losses.append(metrics['loss'])
                train_accs.append(metrics['accuracy'])
            
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)
            
            # Validation
            val_losses = []
            val_accs = []
            
            for data, labels in val_data:
                metrics = self.evaluate(data, labels)
                val_losses.append(metrics['loss'])
                val_accs.append(metrics['accuracy'])
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_acc'].append(avg_val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}"
                )
            
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        return self.history


# Attack type labels for reference
ATTACK_LABELS = {
    0: 'Normal',
    1: 'DoS',           # Denial of Service
    2: 'Probe',         # Reconnaissance/Scanning
    3: 'R2L',           # Remote to Local
    4: 'U2R',           # User to Root (Privilege Escalation)
    5: 'Botnet',        # Botnet Command & Control
    6: 'Lateral',       # Lateral Movement
    7: 'C2',            # Command and Control
    8: 'Exfiltration',  # Data Exfiltration
    9: 'APT'            # Advanced Persistent Threat
}


def create_gnn_detector(
    pretrained_path: Optional[str] = None,
    device: str = 'cpu',
    **kwargs
) -> GNNIntrusionDetector:
    """
    Factory function to create a GNN detector.
    
    Args:
        pretrained_path: Path to pretrained weights
        device: Device to use
        **kwargs: Model configuration
        
    Returns:
        Configured GNN detector
    """
    model = GNNIntrusionDetector(**kwargs)
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained GNN model from {pretrained_path}")
    
    return model.to(device)


if __name__ == "__main__":
    # Demo usage
    print("GNN Intrusion Detector Demo")
    print("=" * 50)
    
    # Create model
    model = GNNIntrusionDetector(
        node_features=32,
        edge_features=16,
        hidden_dim=128,
        num_classes=10,
        num_gat_layers=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample graph
    graph_builder = NetworkGraphBuilder()
    
    # Simulate some network flows
    from datetime import datetime
    now = datetime.now()
    
    for i in range(100):
        graph_builder.add_flow(
            src_ip=f"192.168.1.{i % 20}",
            dst_ip=f"10.0.0.{i % 10}",
            src_port=50000 + i,
            dst_port=80 if i % 3 == 0 else 443,
            protocol="TCP",
            bytes_sent=1000 * (i + 1),
            bytes_recv=500 * (i + 1),
            packets=10 + i,
            duration=1.5,
            timestamp=now
        )
    
    # Build graph
    data = graph_builder.build_graph()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {data.x.size(0)}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Node features: {data.x.size(1)}")
    
    # Forward pass
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr, return_embeddings=True)
    
    print(f"\nModel Output:")
    print(f"  Node predictions shape: {output['node_logits'].shape}")
    print(f"  Graph predictions shape: {output['graph_logits'].shape}")
    print(f"  Node embeddings shape: {output['node_embeddings'].shape}")
    
    # Compute anomaly scores
    anomaly_scores = model.compute_anomaly_score(output)
    print(f"\nAnomaly Scores:")
    print(f"  Mean: {anomaly_scores.mean():.4f}")
    print(f"  Max: {anomaly_scores.max():.4f}")
    print(f"  High anomaly nodes: {(anomaly_scores > 0.7).sum()}")
    
    print("\n✅ GNN Detector ready for network intrusion detection!")
