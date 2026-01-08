import torch

def get_train_dataloaders(dataset, batch_size=128, num_workers=20, **kwargs):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        **kwargs
    )

def get_test_val_dataloader(dataset, batch_size=128, num_workers=2, drop_last=False, shuffle=False, **kwargs):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        **kwargs
    )
