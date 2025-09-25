from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import numpy as np
from train import train_model

def cross_validate(model_fn, dataset, k, batch_size, device, **train_kwargs):
    #model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device, accu=False
    """
    model_fn: function that returns a new instance of your model
    dataset: full dataset (torch.utils.data.Dataset)
    k: number of folds
    batch_size: batch size for DataLoaders
    device: 'cuda' or 'cpu'
    train_kwargs: arguments for train_model()
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_all = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n===== Fold {fold+1}/{k} =====")

        # Make subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # New model per fold
        model = model_fn().to(device)

        # Train
        metrics = train_model(
            model=model,
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            device=device,
            **train_kwargs
        )

        metrics_all.append(metrics)

    # Average metrics across folds
    avg_metrics = {}
    for key in metrics_all[0].keys():
        avg_metrics[key] = np.mean([m[key][-1] for m in metrics_all])  # last epoch metric

    print("\n===== Cross-validation results =====")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics_all, avg_metrics
