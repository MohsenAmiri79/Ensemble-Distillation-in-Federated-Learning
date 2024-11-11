def federated_avg(global_model, delta_dict):
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] += delta_dict[k] / len(delta_dict)

    global_model.load_state_dict(global_dict)

    return global_model
