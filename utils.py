import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def compute_loss(pos_score, neg_score, device):
    """
    Computes Binary Cross Entropy Loss with Logits for positive and negative graph edge scores.
    """
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def evaluation(pos_score, neg_score):
    """
    Evaluates model performance by calculating Accuracy, F1-Score, and ROC-AUC.
    """
    # Convert tensors to CPU numpy arrays
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    
    # Apply a standard threshold of 0.5 to get binary predictions
    preds = (scores >= 0.5).astype(int)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
        
    return acc, f1, auc

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Creates a learning rate schedule that increases linearly during a 'warmup' phase,
    then decreases following a cosine curve.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * num_cycles * 2.0 * progress)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
