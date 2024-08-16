import torch

GT = torch.load("/data/users/mpilligua/hashing-nvd/results/summer/lucia_lucia/test.pth")
Pred = torch.load("/data/users/mpilligua/hypersprites/pred_params.pth")

# print(GT["model_F_mappings_state_dict"].keys())
# print("\n\n")
# print([(k,a.shape) for k, a in Pred.items()])


gt2pred = {"model_texture.texture.network":"texture_net", "model_mapping.network":"mapping_net", "model_residual.residual.network":"residual_net"}
pred2gt = {v:k for k, v in gt2pred.items()}

for k, v in GT["model_F_mappings_state_dict"].items():
    global_layer = k.split(".")[0]
    local_layer = int(k.split(".")[-2]) // 2
    net_name = ".".join(k.split(".")[1:-2])
    
    if "bias" in k:
        continue
        
    if net_name in gt2pred:
        if "params" in k:
            net = net_name.split(".")[1]
            if net in ["texture", "residual"]:
                new_name = f"layer_{global_layer}.{net}_enc.embeddings.weight"
        else:
            new_name = f"layer_{global_layer}.{gt2pred[net_name]}.layer_{local_layer}.weight"

        print(k, v.shape)
        print(new_name, Pred[new_name].shape)
        
        # if v.shape == Pred[new_name].squeeze(0).shape:
        #     print("Equal")
        # else: 
        #     print("Not equal")
        # print("\n")
        
    else: 
        print("Not in mapping", k)