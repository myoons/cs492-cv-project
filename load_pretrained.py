import torch
import os

def load_best_model(path, num_classes, args):    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    model = """Model Name"""(num_classes)
    bestWeight = torch.load(path, map_location=args.device)
    model.load_state_dict(bestWeight)

    return model

model = load_best_model("Path of .pt File", "num_classes", "args")