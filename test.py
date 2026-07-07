import torch
from graph import GraphCreateK, UnDirectionalGraphCreateK
from model import GraphSAGE, DotPredictor, MLPPredictor, LinearPredictor, GAT, GCN
from utils import evaluation
from aggregate_models import LSTMAggregator, MeanAggregator, MaxAggregator
import argparse
import matplotlib.pyplot as plt
import numpy as np

model_names = ['lstm', 'max', 'mean']
test_accs = []
test_f1s = []
test_aucs = []
metrics = ['Accuracy', 'F1 Score', 'AUC']

x = np.arange(len(model_names))
width = 0.25

AGGREGATOR_CONSTRUCTOR = {
    'lstm': LSTMAggregator,
    'mean': MeanAggregator,
    'max': MaxAggregator
}

GNN_CONSTRUCTOR = {
    'sage': GraphSAGE,
    'gat': GAT,
    'gcn': GCN
}

DECISION_MODEL = {
    'dot': DotPredictor,
    'mlp': MLPPredictor,
    'lin': LinearPredictor
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model1', default='model/lstm_sage_dot_testIdx3_best.pkl')
    ap.add_argument('--model2', default='model/max_sage_dot_testIdx3_best.pkl')
    ap.add_argument('--model3', default='model/mean_sage_dot_testIdx3_best.pkl')
    ap.add_argument('--device', default='cuda:0')
    return ap.parse_args()

def main():
    cli_args = parse_args()
    models = [cli_args.model1, cli_args.model2, cli_args.model3]
    device = cli_args.device if torch.cuda.is_available() else "cpu"
    
    for i in range(3):
        data = torch.load(models[i], map_location=device)
        train_args = data['args']
        print(f"Evaluating Config for model: {model_names[i]}")
        print(train_args)
        
        n_data_feature = data['feature']
        
        if train_args.bidirectional:
            train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = UnDirectionalGraphCreateK(train_args)
        else:
            train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = GraphCreateK(train_args)
            
        train_g = train_g.to(device)
        train_pos_g, train_neg_g = train_pos_g.to(device), train_neg_g.to(device)
        valid_pos_g, valid_neg_g = valid_pos_g.to(device), valid_neg_g.to(device)
        test_pos_g, test_neg_g = test_pos_g.to(device), test_neg_g.to(device)
        
        n_data_feature = n_data_feature.to(device)
        
        model = GNN_CONSTRUCTOR[train_args.gnn](train_args.FeatureDim, train_args.HiddenLayerDim)
        model.load_state_dict(data['gnn'])
        model = model.to(device)
        model.eval()
        
        pred = DECISION_MODEL[train_args.Predictor](train_args.HiddenLayerDim)
        pred.load_state_dict(data['pred'])
        pred.to(device)
        pred.eval()
        
        with torch.no_grad():
            h = model(train_g, n_data_feature)
            pos_score = torch.sigmoid(pred(test_pos_g, h))
            neg_score = torch.sigmoid(pred(test_neg_g, h))
            test_acc, test_f1, test_auc = evaluation(pos_score, neg_score)
            
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
        print('test acc: {:.4f} f1: {:.4f} auc: {:.4f}\n'.format(test_acc, test_f1, test_auc))
        
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, test_accs, width, label='Accuracy', color='skyblue')
    plt.bar(x, test_f1s, width, label='F1 Score', color='lightgreen')
    plt.bar(x + width, test_aucs, width, label='AUC', color='salmon')
    
    # Labels and legend
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on bars
    def add_labels(data_list, offset):
        for idx in range(len(data_list)):
            plt.text(x[idx] + offset, data_list[idx] + 0.01, f'{data_list[idx]:.3f}', ha='center', va='bottom', fontsize=9)
            
    add_labels(test_accs, -width)
    add_labels(test_f1s, 0)
    add_labels(test_aucs, width)
    
    plt.tight_layout()
    plt.savefig("test_metrics.png")
    plt.show()

if __name__ == '__main__':
    main()
