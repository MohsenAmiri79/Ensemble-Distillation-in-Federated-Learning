import torch
from .lenet import LeNet
from .datamodule import create_pseudo_loader
from .train import train_model


def get_client_logits(global_model_dict, delta_dict, dataloader, conf):
    device = conf['device']

    for k in delta_dict.keys():
        if torch.isnan(delta_dict[k]).any():
            print(f"NaN detected in delta_dict for key {k}")
            break

    client_model = LeNet().to(device)
    client_model.load_state_dict(global_model_dict)

    client_model_dict = client_model.state_dict()
    for k in delta_dict.keys():
        client_model_dict[k] += delta_dict[k].to(device)
    client_model.load_state_dict(client_model_dict)

    client_model.eval()

    all_logits = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            outputs = client_model(data)
            all_logits.append(outputs.cpu())

    # Concatenate all logits
    all_logits = torch.cat(all_logits, dim=0)

    return all_logits


def ensemble_distillation(delta_dicts, global_model_dict,
                          server_dataloader, conf):
    logits_list = []
    for delta_dict in delta_dicts:
        logits = get_client_logits(global_model_dict, delta_dict,
                                   server_dataloader,
                                   {'device': conf['device']})
        logits_list.append(logits)

    averaged_logits = torch.mean(torch.stack(logits_list), dim=0)
    pseudo_dataloader = create_pseudo_loader(
        server_dataloader, averaged_logits, conf)

    return train_model(global_model_dict, pseudo_dataloader,
                       conf, global_model=True)
