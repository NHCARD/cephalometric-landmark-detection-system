from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import UNet as network
from utils.dataload import dataload_train

H = 224
W = 224
pow_n = 10
batch_size = 4
num_workers = 5
dataloaders = {
    'train': DataLoader(dataload_train(path=r"./CL-Data/dataset/train", H=H, W=W, pow_n=pow_n, aug=True), batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'valid': DataLoader(dataload_train(path=r"./CL-Data/dataset/val", H=H, W=W, pow_n=pow_n, aug=False), batch_size=batch_size, shuffle=False, num_workers=num_workers)
}


def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2))
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 38

if __name__ == '__main__':
    model = network.UNet(1, num_classes).to(device)

    num_epochs = 2000
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    print("****************************GPU : ", device)

    best_loss = 1e10

    for epoch in range(1, num_epochs + 1):
        print('========================' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('------------------------' * 10)

        phases = ['train', 'valid'] if epoch % 10 == 0 else ['train']

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0
            pbar = tqdm.tqdm(dataloaders[phase], unit='batch')

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    LOSS = L2_loss(outputs, labels)
                    metrics['Jointloss'] += LOSS

                    if phase == 'train':
                        LOSS.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            if epoch % 5 == 0:
                pred = outputs[0].cpu().detach().numpy()
                plt.imshow(pred[0], cmap='gray')
                plt.show()

            epoch_Jointloss = metrics['Jointloss'] / epoch_samples

            for param_group in optimizer.param_groups:
                lr_rate = param_group['lr']
            print(phase, "Joint loss :", epoch_Jointloss.item(), 'lr rate', lr_rate)

            savepath = 'model/net_{}_E_{}.pt'
            if phase == 'valid' and epoch_Jointloss < best_loss:
                print("model saved")
                best_loss = epoch_Jointloss
                torch.save(model.state_dict(), savepath.format(best_loss,  epoch))
