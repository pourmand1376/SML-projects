import torch
from conv       import Conv2d
from linear     import Linear
from sequential import Sequential

conversion_table = { 
        'Linear': Linear,
        'Conv2d': Conv2d
    }

def convert_vgg(module, modules=None):
    # First time
    if modules is None: 
        modules = []
        for m in module.children():
            convert_vgg(m, modules=modules)


            if isinstance(m, torch.nn.AdaptiveAvgPool2d): 
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential): 
        for m in module.children():
            convert_vgg(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)

    elif isinstance(module, torch.nn.ReLU): 

        modules.append(torch.nn.ReLU())
    else:
        modules.append(module)

