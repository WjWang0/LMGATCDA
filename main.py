import time
from torch.nn import BCEWithLogitsLoss
from dgl import NID, EID
from dgl.dataloading import GraphDataLoader
from utils import *
from sampler import SEALData
from model import *
from sklearn.model_selection import StratifiedKFold
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train(model, dataloader, loss_fn, optimizer, device, num_graphs=32, total_graphs=None):
    model.train()

    total_loss = 0
    for g, labels in dataloader:
        g = g.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(g, g.ndata['z'], g.ndata[NID], g.edata[EID])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs

    return total_loss / total_graphs,logits,labels


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    y_pred, y_true = [], []
    for g, labels in dataloader:
        g = g.to(device)
        labels = labels.to(device)
        logits = model(g, g.ndata['z'], g.ndata[NID], g.edata[EID])
        loss = loss_fn(logits, labels)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(labels.view(-1).cpu().to(torch.float))

    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    pos_pred = y_pred[y_true == 1]
    neg_pred = y_pred[y_true == 0]

    return pos_pred, neg_pred, loss


def main(args,  print_fn=print, alpha_num = None, hop_num = None ):
    print_fn("Experiment arguments: {}".format(args))
    print_fn("alpha:{}    hop_num:{}".format(alpha_num, hop_num))

    # set gpu
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu_id)
    else:
        device = 'cpu'

    if args.random_seed:
        torch.manual_seed(args.random_seed)
    else:
        torch.manual_seed(123)
    # Load dataset
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    recall_result = []
    prcs = []
    aucs = []
    acc_result = []
    f1_result = []
    pre_result = []
    graph, IC, ID = build_graph(device = device, directory = args.load_dir)
    fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True,random_state=args.random_seed)
    for train_idx,test_idx in kf.split(X=np.arange(int(graph.num_edges()/2)), y=np.ones(int(graph.num_edges()/2))):
        fold += 1
        split_edge = get_split_edge(g=graph, train_idx=train_idx, test_idx=test_idx)

        seal_data = SEALData(args=args, g=graph, split_edge=split_edge, hop=args.hop, neg_samples=args.neg_samples,
                             subsample_ratio=args.subsample_ratio, use_coalesce=False, prefix='cda',
                             save_dir=args.save_dir, num_workers=args.num_workers, print_fn=print_fn, device = device)


        train_data = seal_data('train')
        val_data = seal_data('valid')
        test_data = seal_data('test')

        train_graphs = len(train_data.graph_list)

        train_loader = GraphDataLoader(train_data, batch_size=args.batch_size,num_workers=args.num_workers)
        val_loader = GraphDataLoader(val_data,  batch_size=args.batch_size,num_workers=args.num_workers)
        test_loader = GraphDataLoader(test_data,batch_size=args.batch_size, num_workers=args.num_workers)

        model = Model(args=args,
                      G=graph,
                      alpha=alpha_num,
                      hop_num=hop_num,
                      model_name = args.model,
                      num_diseases=IC.shape[0],
                      num_rna=ID.shape[0],
                      num_layers=args.num_layers,
                      in_dim=args.in_dim,
                      hidden_units = args.hidden_units,
                      gnn_type = args.gnn_type,
                      dropout = args.dropout,
                      device=device,
                      use_attribute=args.use_attribute)


        model = model.to(device)
        parameters = model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        loss_fn = BCEWithLogitsLoss()
        print_fn("Total parameters: {}".format(sum([p.numel() for p in model.parameters()])))
        start_time = time.time()


        for epoch in range(args.epochs):

            loss, logits, labels = train(model=model,
                         dataloader=train_loader,
                         loss_fn=loss_fn,
                         optimizer=optimizer,
                         device=device,
                         num_graphs=args.batch_size,
                         total_graphs=train_graphs)
            train_time = time.time()
            val_pos_pred, val_neg_pred, val_loss= evaluate(model=model,
                                                           dataloader=val_loader,
                                                           loss_fn = loss_fn,
                                                           device=device)
            val_acc, val_pre, val_recall,precision_, recall_, val_f1, val_auc, val_prc, fpr, tpr = evaluate_roc(val_pos_pred, val_neg_pred)
            print_fn(
                "Epoch-{}, train loss: {:.4f}, val_auc-{:.4f},val_acc:{:.4f},val_pre-{:.4f},val_recall-{:.4f}, val_f1-{:.4f},val_prc-{:.4f},val_auc-{:.4f}, cost time: train-{:.1f}s".format(
                    epoch, loss, val_auc, val_acc, val_pre, val_recall,val_f1, val_prc, val_auc,train_time - start_time))

        test_pos_pred, test_neg_pred, test_loss = evaluate(model=model,
                                                    dataloader=test_loader,
                                                    loss_fn=loss_fn,
                                                    device=device)
        test_acc, test_pre, test_recall, precision_, recall_, test_f1, test_auc, test_prc, test_fpr, test_tpr = evaluate_roc(
                test_pos_pred, test_neg_pred)

        evaluate_time = time.time()
        print_fn("fold-{}, test loss: {:.4f},test_acc:{:.4f},test_pre-{:.4f},test_recall-{:.4f}, test_f1-{:.4f},test_prc-{:.4f} ,test_auc-{:.4f}"
                         "cost time, total-{:.1f}s".format(fold, test_loss, test_acc, test_pre, test_recall, test_f1,test_prc, test_auc,
                                                                          evaluate_time - start_time))

        aucs.append(test_auc)
        precisions.append(precision_)
        recalls.append(recall_)
        recall_result.append(test_recall)
        pre_result.append(test_pre)
        prcs.append(test_prc)
        acc_result.append(test_acc)

        fprs.append(test_fpr)
        tprs.append(test_tpr)
        f1_result.append(test_f1)


    print_fn("Experiment Results:")
    print_fn("model_type:{}".format(args.gnn_type))
    print_fn("subgraph_ratio:{}".format(args.subsample_ratio))
    print_fn("train_ratio:0.8,val_ratio:0.1,test_ratio:0.1")
    print_fn("hop:{:.4f}".format(args.hop))
    print_fn("Best auc@: {:.4f}, Fold: {}".format(np.max(aucs), np.argmax(aucs)+1))
    print_fn('-----------------------------------------------------------------------------------------------')
    print_fn('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(aucs), np.std(aucs)))
    print_fn('Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)))
    print_fn('Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)))
    print_fn('Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)))
    print_fn('F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)))
    print_fn('PRC mean: %.4f, variace: %.4f \n' % (np.mean(prcs), np.std(prcs)))
    print_fn('fprs:', fprs)
    print_fn('tprs:', tprs)
    print_fn('aucs:', aucs)
    return fprs, tprs, aucs, precisions, recalls, prcs


if __name__ == '__main__':
    args = parse_arguments()
    fprs, tprs, aucs, precisions, recalls, prcs = main(args, alpha_num=args.alpha, hop_num=args.hop_num)

