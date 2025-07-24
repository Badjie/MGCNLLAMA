import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
from transformers import LlamaModel, LlamaConfig

class MGLLAMA(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(MGLLAMA, self).__init__()
        self.hidden_dim = hidden_dim
        self.gcn_input = GCNConv(input_dim, hidden_dim)
        self.gcn_hidden = GCNConv(hidden_dim, hidden_dim)

        self.llama_config = LlamaConfig(
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=0,
            vocab_size=32000
        )
        self.llama = LlamaModel(self.llama_config)
        self.linear_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing_info(self, h_v, h_N_v):
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        beta = torch.tanh(self.W_beta1(h_v) + self.W_beta2(h_N_v))
        r = torch.zeros_like(h_v)
        r_v = (gamma + 1) * r + beta
        m_v = h_v + r_v - h_N_v
        return m_v

    def forward(self, x, edge_index, h_c, batch=None):
        x_gnn = self.gcn_input(x, edge_index)
        x_gnn = self.dropout(x_gnn)
        h_N_v = self.gcn_hidden(x_gnn, edge_index)
        x_gnn = self.predict_missing_info(x_gnn, h_N_v)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        padded_x = torch.zeros((num_graphs, max_nodes, self.hidden_dim), device=x.device)

        for i in range(num_graphs):
            node_indices = (batch == i).nonzero(as_tuple=True)[0]
            padded_x[i, :len(node_indices), :] = x_gnn[node_indices]

        llama_output = self.llama(inputs_embeds=padded_x).last_hidden_state
        x_llama_flat = torch.cat([llama_output[i, :torch.sum(batch == i)] for i in range(num_graphs)], dim=0)

        if x_llama_flat.shape[0] != x_gnn.shape[0]:
            raise ValueError(f"Shape mismatch: x_gnn={x_gnn.shape}, x_llama={x_llama_flat.shape}")

        combined = torch.cat([x_gnn, x_llama_flat], dim=-1)
        output = self.linear_combine(combined)
        output = self.dropout(output)

        if h_c is None:
            h = output
            c = torch.zeros_like(h)
        else:
            h, c = h_c
            h = output

        return h, c

class EdgeClassifier(nn.Module):
    def __init__(self, node_embedding_dim):
        super(EdgeClassifier, self).__init__()
        edge_embedding_dim = 2 * node_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim, edge_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_embedding_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, edge_embeddings):
        return self.mlp(edge_embeddings)

class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EnhancedTemporalGraphNetwork, self).__init__()
        self.mgllama = MGLLAMA(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.edge_classifier = EdgeClassifier(hidden_dim)

    def create_edge_embeddings(self, node_embeddings, edge_index):
        if isinstance(node_embeddings, tuple):
            node_embeddings = node_embeddings[0]
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return edge_embeddings

    def forward(self, x, edge_index, batch=None):
        h = None
        for _ in range(self.num_layers):
            h, c = self.mgllama(x, edge_index, h, batch)
            h = (h, c)

        edge_embeddings = self.create_edge_embeddings(h[0], edge_index)
        edge_predictions = self.edge_classifier(edge_embeddings)
        return edge_predictions, h
    
def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['timestamp'])
    return G

def calculate_link_and_antilink_scores(G):
    fraud_scores = nx.out_degree_centrality(G)
    antifraud_scores = nx.eigenvector_centrality(G, max_iter=1000)
    return fraud_scores, antifraud_scores

def label_nodes(fraud_scores, antifraud_scores, fraud_threshold=0.01, antifraud_threshold=0.001):
    labels = {}
    for node in fraud_scores:
        set_label = 'set_usual' if fraud_scores[node] > fraud_threshold else 'set_unusual'
        payment_label = 'payment_usual' if antifraud_scores[node] > antifraud_threshold else 'payment_unusual'
        labels[node] = (set_label, payment_label)
    return labels

def random_walk(G, start_node, num_steps=7):
    current_node = start_node
    visited_nodes = {current_node}
    for _ in range(num_steps):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        visited_nodes.add(current_node)
    return visited_nodes

def label_edges_with_random_walk(G, num_steps=7):
    random_walk_subgraphs = defaultdict(nx.DiGraph)
    for node in G.nodes:
        walk_nodes = random_walk(G, node, num_steps)
        for u in walk_nodes:
            for v in walk_nodes:
                if G.has_edge(u, v):
                    random_walk_subgraphs[node].add_edge(u, v, weight=G[u][v]['weight'])
    return random_walk_subgraphs

def count_edges(random_walk_subgraph, label):
    count = 0
    for u, v, data in random_walk_subgraph.edges(data=True):
        if label in data:
            count += 1
    return count

def extract_features_with_random_walk(G, node, num_steps=7):
    random_walk_subgraphs = label_edges_with_random_walk(G, num_steps)
    neighbors = list(random_walk_subgraphs[node].nodes)
    
    T1 = count_edges(random_walk_subgraphs[node], label='set_usual')
    T2 = count_edges(random_walk_subgraphs[node], label='set_unusual')
    T3 = count_edges(random_walk_subgraphs[node], label='payment_usual')
    T4 = count_edges(random_walk_subgraphs[node], label='payment_unusual')
    
    node_features = [T1, T2, len(neighbors), 
                     T3, T4, len(neighbors), 
                     G.in_degree(node), G.out_degree(node)]
    return node_features

def create_edge_labels(G, labels, edge_index, node_to_idx):
    edge_labels = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()
        src_node = list(G.nodes())[src_idx]
        dst_node = list(G.nodes())[dst_idx]
        src_label = 1 if labels[src_node][0] == 'set_unusual' else 0
        dst_label = 1 if labels[dst_node][0] == 'set_unusual' else 0
        edge_labels.append(float(src_label or dst_label))
    return torch.tensor(edge_labels, dtype=torch.float)

def weighted_cross_entropy_loss(predictions, targets, pos_weight):
    epsilon = 1e-7
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    loss = -(pos_weight * targets * torch.log(predictions) + 
             (1 - targets) * torch.log(1 - predictions))
    return loss.mean()



def create_data_list(G_list, labels_list):
    data_list = []
    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
        edge_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) 
                                  for u, v in G.edges], dtype=torch.long).t().contiguous()
        
        x = torch.tensor([extract_features_with_random_walk(G, node) for node in G.nodes], 
                        dtype=torch.float)
        
        edge_labels = create_edge_labels(G, labels, edge_index, node_to_idx)
        data = Data(x=x, edge_index=edge_index, y=edge_labels)
        data_list.append(data)
    
    return data_list

def train_model(model, train_loader, epochs=10, learning_rate=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        for data in train_loader:
            optimizer.zero_grad()
            edge_predictions, _ = model(data.x, data.edge_index, data.batch)
            loss = weighted_cross_entropy_loss(edge_predictions, data.y, pos_weight=1.0)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            edge_predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            all_predictions.append(edge_predictions.squeeze().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    binary_preds = (all_predictions > 0.5).astype(int)

    auc_score = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)

    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Usual", "Unusual"], yticklabels=["Usual", "Unusual"])
    plt.xlabel("Predicted", fontsize=22)
    plt.ylabel("True", fontsize=22)
    plt.title("Confusion Matrix of BTA Dataset", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate", fontsize=22)
    plt.ylabel("True Positive Rate", fontsize=22)
    plt.title("ROC Curve", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='purple')
    plt.xlabel("Recall", fontsize=22)
    plt.ylabel("Precision", fontsize=22)
    plt.title("Precision-Recall Curve", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()

    return auc_score, precision, recall, f1

def main():
    file_path = 'E:/ansu/ERC20-stablecoins/soc-sign-bitcoinotc.csv'
    data = pd.read_csv(file_path)

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = data.sort_values(by='timestamp')
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31
    combined_time_slice = max(data['time_slice']) - 1
    data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    # Verify the new distribution of entries across time slices
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()

    # Display the new distribution
    print(new_time_slice_counts)

    G_list = []
    labels_list = []
    
    for time_slice in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == time_slice]
        G = create_graph(slice_data)
        link_scores, antilink_scores = calculate_link_and_antilink_scores(G)
        labels = label_nodes(link_scores, antilink_scores)
        G_list.append(G)
        labels_list.append(labels)
    
    if len(G_list) > 2:
        train_G, temp_G, train_labels, temp_labels = train_test_split(G_list, labels_list, test_size=0.4, shuffle=False)
        val_G, test_G, val_labels, test_labels = train_test_split(temp_G, temp_labels, test_size=0.5, shuffle=False)
    else:
        train_G, train_labels = G_list, labels_list
        val_G, val_labels, test_G, test_labels = [], [], [], []
    
    train_data_list = create_data_list(train_G, train_labels)
    val_data_list = create_data_list(val_G, val_labels) if val_G else []
    test_data_list = create_data_list(test_G, test_labels) if test_G else []
    
    train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False) if val_data_list else None
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False) if test_data_list else None
    
    model = EnhancedTemporalGraphNetwork(input_dim=8, hidden_dim=16, num_layers=2)
    trained_model = train_model(model, train_loader)
    
    if test_loader:
        auc_score, precision, recall, f1 = evaluate_model(trained_model, test_loader)
        print(f"Test AUC: {auc_score:.4f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
    else:
        print("Not enough data to create a test set.")
    
    return trained_model

if __name__ == "__main__":
    trained_model = main()
