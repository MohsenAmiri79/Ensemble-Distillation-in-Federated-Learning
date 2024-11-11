import torch


def evaluate_model(global_model, test_loader, roundidx, conf):
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(conf['device']), target.to(conf['device'])
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the global model at round {
          roundidx+1}: {accuracy:.2f}%')
