import torch
from .lenet import LeNet


def train_model(model_dict, local_dataloader,
                conf, global_model=False):
    model = LeNet().to(conf['device'])
    model.load_state_dict(model_dict)
    # Define loss function and optimizer
    if global_model:
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=conf['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=conf['epochs'])

    # Training loop
    model.train()
    for _ in range(conf['epochs']):
        for data, target in local_dataloader:
            data, target = data.to(conf['device']), target.to(conf['device'])
            optimizer.zero_grad()
            output = model(data)
            if global_model:
                output_softmax = torch.nn.functional.log_softmax(output, dim=1)
                target_softmax = torch.nn.functional.softmax(target, dim=1)
                loss = criterion(output_softmax, target_softmax)
            else:
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

    for k, v in model.state_dict().items():
        if torch.isnan(v).any():
            print(f"NaN detected in global_model after federated_avg for key {
                  k} in round {round+1}")
            break

    # Return the updated model state dict
    return model.state_dict()


def client_training(client_fraction_loaders, global_model_dict,
                    fraction_indices, conf):
    device = conf['device']

    # Initialize delta dicts
    delta_sum_dict = {k: torch.zeros_like(
        v, device=device) for k, v in global_model_dict.items()}
    delta_dicts = []
    for idx, client_loader in enumerate(client_fraction_loaders):
        # Train local model
        local_state_dict = train_model(global_model_dict, client_loader, conf)

        # Add current client's gradients
        delta_dict = {k: torch.zeros_like(
            v, device=device) for k, v in global_model_dict.items()}
        for k in delta_sum_dict.keys():
            delta_sum_dict[k] += (local_state_dict[k] - global_model_dict[k])
            delta_dict[k] = local_state_dict[k]
        delta_dicts.append(delta_dict)
        print(f"Client {fraction_indices[idx]} training complete!")

    return delta_sum_dict, delta_dicts
