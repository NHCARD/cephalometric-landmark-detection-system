import sys
import numpy as np
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from net_hand import *
import net_hand as network
import torch
from func import argsoftmax


class MyWindow(QWidget):
    def __init__(self):
        self.device_txt = 'cuda:0'
        # print(self.device_txt)
        self.test_data = None
        self.H = 800
        self.W = 640
        self.model = network.UNet(1, 38).to(self.device_txt)
        self.model.load_state_dict(torch.load(r"D:\ISBI\model\net_1.3920683841206483e-06_E_709.pth", map_location=self.device_txt))
        self.model = self.model.eval()

        super().__init__()
        self.fileDir = None
        self.initUI()
        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 600)

    def initUI(self):
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        cb = QPushButton("Load Img")
        cb.clicked.connect(self.img_load)
        edit_btn = QPushButton('Edit Scatter')
        edit_btn.clicked.connect(self.edit_scatter)
        layout.addWidget(cb)
        layout.addWidget(edit_btn)
        self.layout = layout
        # self.onComboBoxChanged(cb.currentText())

    # def doGraph1(self):
    #     x = np.arange(0, 10, 0.5)
    #     y1 = np.sin(x)
    #     y2 = np.cos(x)
    #
    #     self.fig.clear()
    #
    #     ax = self.fig.add_subplot(111)
    #     ax.plot(x, y1, label="sin(x)")
    #     ax.plot(x, y2, label="cos(x)", linestyle="--")
    #
    #     ax.set_xlabel("x")
    #     ax.set_xlabel("y")
    #
    #     ax.set_title("sin & cos")
    #     ax.legend()
    #
    #     self.canvas.draw()

    # def doGraph2(self):
    #     X = np.arange(-5, 5, 0.25)
    #     Y = np.arange(-5, 5, 0.25)
    #     X, Y = np.meshgrid(X, Y)
    #     Z = X ** 2 + Y ** 2
    #
    #     self.fig.clear()
    #
    #     ax = self.fig.gca(projection='3d')
    #     ax.plot_wireframe(X, Y, Z, color='black')
    #     self.canvas.draw()

    def img_load(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(self, "Open Img", r'E:\4차\download/',
                                                      self.tr("Video Files (*.png)"))

        img = cv2.imread(self.fileDir)
        self.orig_H = img.shape[0]
        self.orig_W = img.shape[1]

        self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, aug=False), batch_size=1,
                                    shuffle=False, num_workers=5)

        self.predict()

    def predict(self):
        Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
        Ymap, Xmap = torch.tensor(Ymap.flatten(), dtype=torch.float).unsqueeze(1).to(self.device_txt), \
            torch.tensor(Xmap.flatten(), dtype=torch.float).unsqueeze(1).to(self.device_txt)
        ind = torch.cat([Xmap / W, Ymap / H], dim=1)

        for inputs, size, name in self.test_data:
            inputs = inputs.to(self.device_txt)
            # size = size.to(self.device_txt)
            # print(size[0])
            # print(size[1])
            # print(size[2])

            outputs = self.model(inputs)
            # print("outputs.shape : ", outputs.shape)
            pred = torch.cat([argsoftmax(outputs[0].view(-1, H * W), Ymap, beta=1e-3) * (self.orig_H / H),
                              argsoftmax(outputs[0].view(-1, H * W), Xmap, beta=1e-3) * (self.orig_W / W)],
                             dim=1).detach().cpu()
            # print("pred : ", pred)

            inputs = cv2.resize(inputs[0][0].detach().cpu().numpy(), (2880, 2400))

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(inputs, cmap='gray')

            pred = pred.detach().cpu().numpy()

            for i in pred:
                ax.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b')

            self.canvas.draw()

    def edit_scatter(self):
        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
