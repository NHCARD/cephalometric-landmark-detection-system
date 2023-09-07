import sys
import time

import numpy as np
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from net_hand import *
import net_hand as network
import torch
from func import argsoftmax
import subprocess


class MyWindow(QWidget):
    def __init__(self):
        self.layout = None
        self.canvas2 = None
        self.canvas = None
        self.vis_origin = None
        self.vis_output = None
        self.device_txt = 'cuda:0'
        self.test_data = None
        self.H = 800
        self.W = 640
        self.model = network.UNet(1, 38).to(self.device_txt)
        self.model.load_state_dict(torch.load(r"./net_1.3920683841206483e-06_E_709.pth", map_location=self.device_txt))
        self.model = self.model.eval()

        super().__init__()
        self.fileDir = None
        self.initUI()
        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 600)  # 창의 위치 x좌표, y좌표, 가로크기, 세로 크기

    def initUI(self):
        self.vis_output = plt.Figure()  # 모델 아웃풋 이미지
        self.vis_origin = plt.Figure()  # 모델 인풋 이미지
        self.canvas = FigureCanvas(self.vis_output)  # 아웃풋 이미지 출력 박스
        self.canvas2 = FigureCanvas(self.vis_origin)  # 인풋 이미지 출력 박스

        layout = QHBoxLayout()  # 메인 레이아웃 (Horizen)
        layout.addWidget(self.canvas2)
        layout.addWidget(self.canvas)

        btn_layout = QVBoxLayout()  # 버튼, 리스트 레이아웃
        load_img_btn = QPushButton("Load Img")
        load_img_btn.clicked.connect(self.img_load)
        edit_btn = QPushButton('Edit Scatter')
        edit_btn.clicked.connect(self.edit_scatter)

        file_list = QListWidget(self)

        btn_layout.addWidget(load_img_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(file_list)

        layout.addLayout(btn_layout)

        self.layout = layout

    def img_load(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(self, "Open Img", r'./0img',
                                                      self.tr("Video Files (*.png)"))

        if self.fileDir != '':
            img = cv2.imread(self.fileDir)
            self.orig_H = img.shape[0]
            self.orig_W = img.shape[1]

            self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, aug=False), batch_size=1,
                                        shuffle=False, num_workers=5)

            s_time = time.time()
            self.predict()
            e_time = time.time()
            print(e_time - s_time)
        else:
            pass

    def predict(self):
        Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
        Ymap, Xmap = torch.tensor(Ymap.flatten(), dtype=torch.float).unsqueeze(1).to(self.device_txt), \
            torch.tensor(Xmap.flatten(), dtype=torch.float).unsqueeze(1).to(self.device_txt)
        ind = torch.cat([Xmap / W, Ymap / H], dim=1)

        with torch.no_grad():
            for inputs, size, name in self.test_data:
                inputs = inputs.to(self.device_txt)

                outputs = self.model(inputs)
                # print("outputs.shape : ", outputs.shape)
                pred = torch.cat([argsoftmax(outputs[0].view(-1, H * W), Ymap, beta=1e-3) * (self.orig_H / H),
                                argsoftmax(outputs[0].view(-1, H * W), Xmap, beta=1e-3) * (self.orig_W / W)],
                                dim=1).detach().cpu()
                # print("pred : ", pred)

                self.inputs = cv2.resize(inputs[0][0].detach().cpu().numpy(), (2880, 2400))

                self.vis_output.clear()
                ax = self.vis_output.add_subplot(111)
                ax.axis('off')
                ax.imshow(self.inputs, cmap='gray')
                # ax.subplots_adjust(left=0)

                pred = pred.detach().cpu().numpy()

                for i in pred:
                    ax.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b')

                self.canvas.draw()

    def edit_scatter(self):
        if self.fileDir != None:
            self.vis_output, self.ax = plt.subplots(figsize=(12, 10))
            plt.imshow(self.inputs, cmap='gray')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Interactive Plot')

            self.ax.set_aspect('auto', adjustable='box')

            self.xdata = [0]
            self.ydata = [0]
            self.line, = self.ax.plot(self.xdata, self.ydata)

            cid = plt.connect('button_press_event', self.add_point)
            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.about(self, 'Notice', 'Plese Load Img.')

    def add_point(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            x = event.xdata
            y = event.ydata

            self.xdata.append(x)
            self.ydata.append(y)

            plt.scatter(self.xdata, self.ydata, s=20, marker='.', c='b')
            plt.draw()

        if event.button == 3:
            self.xdata.pop()
            self.ydata.pop()
            self.vis_output.figimage(self.inputs, cmap='gray', resize=True)

            plt.scatter(self.xdata, self.ydata, s=20, marker='.', c='b')
            plt.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
