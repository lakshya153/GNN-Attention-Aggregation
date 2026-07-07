import torch
import itertools
import pickle as pkl
import matplotlib.pyplot as plt
import torch.nn as nn
from argument import argument
from graph import GraphCreateK, UnDirectionalGraphCreateK
from model import GraphSAGE, DotPredictor, MLPPredictor, GAT, GCN, LinearPredictor
from aggregate_models import LSTMAggregator, MeanAggregator, MaxAggregator
from utils import evaluation, compute_loss, get_cosine_schedule_with_warmup

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

def set_torch_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = argument()
    set_torch_random_seed(args.torch_random_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.bidirectional:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = UnDirectionalGraphCreateK(args)
    else:
        train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g = GraphCreateK(args)
        
    train_g = train_g.to(device)
    train_pos_g, train_neg_g = train_pos_g.to(device), train_neg_g.to(device)
    valid_pos_g, valid_neg_g = valid_pos_g.to(device), valid_neg_g.to(device)
    test_pos_g, test_neg_g = test_pos_g.to(device), test_neg_g.to(device)
    
    with open(f'data/{args.CNNFeatureExtractor}_{args.FeatureDim}.pkl', 'rb') as f1:
        gene_feature_dict = pkl.load(f1)
        
    # Key of gene_number_dict is name of gene. Value of gene_number_dict is idx of gene
    with open('data/number_gene.pkl', 'rb') as f2:
        gene_number_dict = pkl.load(f2)
        
    gene_features = dict()
    for gene in gene_feature_dict:
        features = []
        for view in ['lateral', 'ventral', 'dorsal']: 
            if view in gene_feature_dict[gene]:
                for embed in gene_feature_dict[gene][view]:
                    features.append(list(embed))
        gene_features[gene_number_dict[gene]] = torch.tensor(features)
        
    aggregator = AGGREGATOR_CONSTRUCTOR[args.aggr](args.FeatureDim)
    aggregator = aggregator.to(device)
    model = GNN_CONSTRUCTOR[args.gnn](args.FeatureDim, args.HiddenLayerDim)
    model = model.to(device)
    model.train()
    
    pred = DECISION_MODEL[args.Predictor](args.HiddenLayerDim)
    pred.to(device)
    
    optimizer = torch.optim.Adam(
        itertools.chain(aggregator.parameters(), model.parameters(), pred.parameters()), 
        lr=args.lr
    )
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.Epoch * args.warmup_portion), args.Epoch)
    
    train_losses, train_accs, train_f1s, train_aucs = [], [], [], []
    valid_accs, valid_f1s, valid_aucs = [], [], []
    best_acc = 0
    
    for epoch in range(args.Epoch):
        n_data_feature = aggregator(gene_features, device) 
        h = model(train_g, n_data_feature)
        pred.train()
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score, device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.warmup:
            lr_scheduler.step()
            
        with torch.no_grad():
            pred.eval()
            model.eval()
            h = model(train_g, n_data_feature)
            train_acc, train_f1, train_auc = evaluation(torch.sigmoid(pos_score), torch.sigmoid(neg_score))
            
            pos_score_val = pred(valid_pos_g, h)
            neg_score_val = pred(valid_neg_g, h)
            valid_acc, valid_f1, valid_auc = evaluation(torch.sigmoid(pos_score_val), torch.sigmoid(neg_score_val))
            
        if epoch % 2 == 0:
            print('In epoch {}, train loss: {:.4f} acc: {:.4f} f1: {:.4f} auc: {:.4f}'.format(epoch, loss.item(), train_acc, train_f1, train_auc))
            print('valid acc: {:.4f} f1: {:.4f} auc: {:.4f}'.format(valid_acc, valid_f1, valid_auc))
            
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        train_aucs.append(train_auc)
        valid_accs.append(valid_acc) 
        valid_f1s.append(valid_f1)
        valid_aucs.append(valid_auc)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            e = epoch
            torch.save({
                'gnn': model.state_dict(),
                'args': args,
                'pred': pred.state_dict(),
                'aggregator': aggregator.state_dict() if hasattr(aggregator, 'state_dict') else aggregator,
                'feature': n_data_feature.cpu().detach(),
                'epoch': e + 1,
                'epochs': range(epoch + 1)
            }, 'model/%s_%s_%s_testIdx%s_best.pkl' % (args.aggr, args.gnn, args.Predictor, args.test_idx))

    epochs_range = range(len(train_losses))
    plt.figure(figsize=(14, 10))
    
    # Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, valid_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    # F1 Score Plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1')
    plt.plot(epochs_range, valid_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs') 
    plt.legend()
    
    # AUC Plot
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, train_aucs, label='Train AUC')
    plt.plot(epochs_range, valid_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"training_metrics_{args.aggr}.png")
    plt.show()

if __name__ == '__main__':
    main()
