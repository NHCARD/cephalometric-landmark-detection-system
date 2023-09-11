import glob
import os
import sys
import time

import numpy as np
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import mytransforms.functional
from net_hand import *
import net_hand as network
import torch
from func import argsoftmax
import subprocess
import openpyxl

H = 800
W = 640


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
        self.mode = None
        self.model = network.UNet(1, 38).to(self.device_txt)
        self.model.load_state_dict(torch.load(r"./net_1.3920683841206483e-06_E_709.pth", map_location=self.device_txt))
        self.model = self.model.eval()
        self.pred = []
        self.input = []

        super().__init__()
        self.fileDir = None
        self.initUI()
        self.setLayout(self.layout)
        self.setGeometry(200, 200, 1400, 800)  # 창의 위치 x좌표, y좌표, 가로크기, 세로 크기

    def initUI(self):
        self.vis_output = plt.Figure(figsize=(10, 10))  # 모델 아웃풋 이미지
        self.vis_origin = plt.Figure(figsize=(5, 5))  # 모델 인풋 이미지
        self.canvas = FigureCanvas(self.vis_output)  # 아웃풋 이미지 출력 박스
        self.canvas2 = FigureCanvas(self.vis_origin)  # 인풋 이미지 출력 박스

        menubar = QMenuBar(self)
        # menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&Menu')
        # filemenu.

        layout = QHBoxLayout()  # 메인 레이아웃 (Horizon)
        layout.addWidget(self.canvas2)
        layout.addWidget(self.canvas)

        btn_layout = QVBoxLayout()  # 버튼, 리스트 레이아웃
        load_img_btn = QPushButton("Load Img")
        load_img_btn.clicked.connect(self.img_load)
        directory_btn = QPushButton("Load Directory")
        directory_btn.clicked.connect(self.directory_load)
        btn_layout = QVBoxLayout()  # 버튼, 리스트 레이아웃
        load_img_btn = QPushButton("Load Img")
        load_img_btn.clicked.connect(self.img_load)
        edit_btn = QPushButton('Edit Scatter')
        edit_btn.clicked.connect(self.edit_scatter)

        self.file_list = QListWidget(self)
        self.file_list.itemDoubleClicked.connect(self.List_click)
        self.del_list = QPushButton('Delete')
        self.del_list.clicked.connect(self.Delete_list)

        btn_layout.addWidget(load_img_btn)

        btn_layout.addWidget(directory_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(self.file_list)
        btn_layout.addWidget(self.del_list)

        layout.addLayout(btn_layout)

        self.layout = layout

    def img_load(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(self, "Open Img", r'./0img',
                                                      self.tr("Video Files (*.png)"))

        if self.fileDir != '':
            img = cv2.imread(self.fileDir)
            self.orig_H = img.shape[0]
            self.orig_W = img.shape[1]

            find_idx = self.fileDir.rindex('/')
            # print(f'fine_idx : {find_idx}')
            self.file_list.addItem(self.fileDir[find_idx + 1:])

            # img = cv2.resize(img, (800, 640))

            # tf_totensor = mytransforms.ToTensor()
            # tf_resize = mytransforms.Resize((self.H, self.W))

            # print(img.shape)
            # self.test_data = img.reshape((1,) + img.shape)
            # self.test_data = Image.fromarray(self.test_data)
            # self.test_data = tf_totensor(self.test_data)

            self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, aug=False, mode='img'),
                                        batch_size=1,
                                        shuffle=False, num_workers=5)

            s_time = time.time()
            self.predict()
            e_time = time.time()
            print(e_time - s_time)
        else:
            pass

    def directory_load(self):
        self.fileDir = QFileDialog.getExistingDirectory(self, "Open Directory", r'./')

        if self.fileDir != '':
            self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, aug=False, mode='dir'),
                                        batch_size=1,
                                        shuffle=False, num_workers=5)

            f_list = os.listdir(self.fileDir)
            for i in f_list:
                self.file_list.addItem(f'{i}')

            img = cv2.imread(glob.glob(self.fileDir + '/*.png')[0])

            self.orig_H = img.shape[0]
            self.orig_W = img.shape[1]

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

        with torch.no_grad():
            for inputs, size in self.test_data:

                inputs = inputs.to(self.device_txt)

                outputs = self.model(inputs)
                # print("outputs.shape : ", outputs.shape)
                pred = torch.cat([argsoftmax(outputs[0].view(-1, H * W), Ymap, beta=1e-3) * (self.orig_H / H),
                                  argsoftmax(outputs[0].view(-1, H * W), Xmap, beta=1e-3) * (self.orig_W / W)],
                                 dim=1).detach().cpu()
                # print(len(size))

                self.pred.append(pred)
                # print("pred : ", pred)

                self.inputs_resize = cv2.resize(inputs[0][0].detach().cpu().numpy(), (2880, 2400))
                self.input.append(self.inputs_resize)

            self.vis_output.clear()
            self.vis_origin.clear()

            self.pred_plot = self.vis_output.add_subplot(111)
            self.origin_plot = self.vis_origin.add_subplot(111)

            self.pred_plot.axis('off')
            self.pred_plot.imshow(self.inputs_resize, cmap='gray')

            self.origin_plot.axis('off')
            self.origin_plot.imshow(self.inputs_resize, cmap='gray')

            pred = pred.detach().cpu().numpy()

            for idx, i in enumerate(pred):
                self.pred_plot.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b')

                self.pred_plot.text(i[1] + 0.02, i[0] + 0.02, f'{idx + 1}', c='b', fontsize=7)

            self.canvas.draw()
            self.canvas2.draw()

    def edit_scatter(self):
        if self.fileDir != None:
            current_num = self.file_list.currentRow()
            self.edit_output, self.edit_plot = plt.subplots(figsize=(12, 10))
            plt.imshow(self.input[current_num], cmap='gray')

            for y, x in self.pred[current_num]:
                self.edit_plot.scatter(int(x), int(y), s=20, marker='.', c='b')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Interactive Plot')

            self.edit_plot.set_aspect('auto', adjustable='box')

            self.xdata = [0]
            self.ydata = [0]
            self.line, = self.edit_plot.plot(self.xdata, self.ydata)

            cid = plt.connect('button_press_event', self.add_point)
            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.about(self, 'Notice', 'Plese Load IMG.')

    def add_point(self, event):
        if event.inaxes != self.edit_plot:
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
            self.edit_output.figimage(self.inputs, cmap='gray', resize=True)

            plt.scatter(self.xdata, self.ydata, s=20, marker='.', c='b')
            plt.draw()

    def List_click(self):
        if self.fileDir != None:
            num = self.file_list.currentRow()
            # print(self.file_list.currentItem().text())
            # print(num)
            # print(len(self.input))

            self.vis_output.clear()
            self.vis_origin.clear()
            self.pred_plot = self.vis_output.add_subplot(111)
            self.origin_plot = self.vis_origin.add_subplot(111)
            self.origin_plot.axis('off')
            self.pred_plot.axis('off')

            self.pred_plot.imshow(self.input[num], cmap='gray')
            self.origin_plot.imshow(self.input[num], cmap='gray')

            for idx, i in enumerate(self.pred[num]):
                self.pred_plot.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b')
                self.pred_plot.text(i[1] + 0.02, i[0] + 0.02, f'{idx + 1}', c='b', fontsize=7)

            self.canvas.draw()
            self.canvas2.draw()

    def Delete_list(self):
        del_num = self.file_list.currentRow()
        self.file_list.takeItem(del_num)
        self.input.pop(del_num)
        self.pred.pop(del_num)

        # 현재 선택된 표시된 이미지 출력했을 때 다음 이미지로 뜨개 처리

    def excel_save(self):
        self.savedir = QFileDialog.getExistingDirectory(self, "Open Directory", r'./')
        write_wb = openpyxl.Workbook()

        write_ws = write_ws = write_wb.create_sheet('Sheet1')
        write_ws = write_wb.active
        for i in range(1, 38):
            write_ws[f'{str(i)}']
        # write_ws['B1'] = 'Sella'
        write_ws['A2'] = self.file_list.currentItem().text()
        write_ws['B2'] = 0

        write_wb.save('./test.xlsx')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
