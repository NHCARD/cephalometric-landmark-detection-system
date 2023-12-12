import torch
import UNet as network


def argsoftmax(x, index, beta=1e-2):
    a = torch.exp(-torch.abs(x - x.max(dim=1).values.unsqueeze(1)) / beta)
    b = torch.sum(a, dim=1).unsqueeze(1)
    softmax = a / b
    return torch.mm(softmax, index)


def load_model(phase='train', num_classes=38):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = network.UNet(1, num_classes).to(device)
    if phase != 'train':
        model.load_state_dict(torch.load(r"net_1.3920683841206483e-06_E_709.pth", map_location=device))
        model = model.eval()

    return model
