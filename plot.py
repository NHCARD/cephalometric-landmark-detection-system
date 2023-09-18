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
        self.scat_list = []
        self.layout = None
        self.origin_canvas = None
        self.output_canvas = None
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
        self.current_num = None
        self.update_pred = None
        self.landmark_name = []

        super().__init__()
        self.fileDir = None
        self.initUI()
        self.setLayout(self.layout)
        self.setWindowTitle('Landmark Detection System')
        self.setGeometry(200, 200, 1400, 800)  # 창의 위치 x좌표, y좌표, 가로크기, 세로 크기
        self.setFixedWidth(1400)
        self.setFixedHeight(800)

    def initUI(self):
        self.vis_output = plt.Figure(figsize=(10, 10))  # 모델 아웃풋 이미지
        self.vis_origin = plt.Figure(figsize=(5, 5))  # 모델 인풋 이미지

        self.origin_canvas = FigureCanvas(self.vis_origin)  # 인풋 이미지 출력 박스
        self.output_canvas = FigureCanvas(self.vis_output)  # 아웃풋 이미지 출력 박스


        menu_load_img = QAction('&Load Img', self)
        menu_load_img.setShortcut('Ctrl+l')
        menu_load_img.setStatusTip('Load to open Image ')
        menu_load_img.triggered.connect(self.img_load)

        menu_load_dir = QAction('&Load Dir', self)
        menu_load_dir.setShortcut('Ctrl+d')
        menu_load_dir.setStatusTip('Load to Image Directory Folder ')
        menu_load_dir.triggered.connect(self.directory_load)

        menu_edit = QAction('&Edit Scatter', self)
        menu_edit.setShortcut('Ctrl+e')
        menu_edit.setStatusTip('Image Edit to Scatter')
        menu_edit.triggered.connect(self.edit_scatter)

        menu_excel = QAction('&Save Excel', self)
        menu_excel.setShortcut('Ctrl+x')
        menu_excel.setStatusTip('Save Exdcel File')
        menu_excel.triggered.connect(self.excel_save)

        menubar = QMenuBar(self)  # 메뉴바
        fileMenu = menubar.addMenu('&File')

        fileMenu.addAction(menu_load_img)
        fileMenu.addAction(menu_load_dir)
        fileMenu.addAction(menu_edit)
        fileMenu.addAction(menu_excel)
        load_dir = QAction('Load Directory', self)

        layout = QHBoxLayout()  # 메인 레이아웃 (Horizon)
        layout.addWidget(self.origin_canvas)
        layout.addWidget(self.output_canvas)

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
        self.file_list.itemDoubleClicked.connect(self.list_click)

        self.del_list = QPushButton('Delete')
        self.del_list.clicked.connect(self.Delete_list)

        btn_layout.addWidget(load_img_btn)
        btn_layout.addWidget(directory_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(self.file_list)
        btn_layout.addWidget(self.del_list)

        layout.addLayout(btn_layout)
        layout.setMenuBar(menubar)

        self.layout = layout

    def img_load(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(self, "Open Img", r'./0img',
                                                      self.tr("Video Files (*.png)"))

        if self.fileDir != '':
            img = cv2.imread(self.fileDir)
            # sys.exit()

            self.orig_H = img.shape[0]
            self.orig_W = img.shape[1]

            img = Image.open(self.fileDir).convert('L')
            print(img.size)
            img = img.resize((640, 800))
            print(img.size)

            tf_tensor = mytransforms.ToTensor()

            # self.test_data = tf_tensor(img)
            # self.test_data = self.test_data.unsqueeze(dim=0)
            # print(self.test_data.shape)

            find_idx = self.fileDir.rindex('/')
            self.file_list.addItem(self.fileDir[find_idx + 1:])

            self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, aug=False, mode='img'),
                                        batch_size=1,
                                        shuffle=False, num_workers=5)

            s_time = time.time()
            self.predict()
            e_time = time.time()
            print(e_time - s_time)
        else:
            pass

        self.current_num = self.file_list.count() - 1

    def directory_load(self):
        self.fileDir = QFileDialog.getExistingDirectory(self, "Open Directory", r'./')

        if self.fileDir != '':
            self.test_data = DataLoader(dataload(path=self.fileDir, H=self.H, W=self.W, aug=False, mode='dir'),
                                        batch_size=1,
                                        shuffle=False, num_workers=5)

            f_list = os.listdir(self.fileDir)

            for i in f_list:
                self.file_list.addItem(f'{i}')

            self.current_num = self.file_list.count() - 1

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
            for inputs in self.test_data:
                # input = inputs[0][0].numpy()
                # plt.imshow(input)
                # plt.show()
                inputs = inputs.to(self.device_txt)
                # sys.exit()

                outputs = self.model(inputs)
                pred = torch.cat([argsoftmax(outputs[0].view(-1, H * W), Ymap, beta=1e-3) * (self.orig_H / H),
                                  argsoftmax(outputs[0].view(-1, H * W), Xmap, beta=1e-3) * (self.orig_W / W)],
                                 dim=1).detach().cpu()

                self.pred.append(pred)

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

            self.scat_list = []
            self.text_list = []
            for idx, i in enumerate(pred):
                self.scat_list.append(self.pred_plot.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b'))
                self.text_list.append(self.pred_plot.text(i[1] + 0.02, i[0] + 0.02, f'{idx + 1}', c='r', fontsize=7))

            self.output_canvas.draw()
            self.origin_canvas.draw()

    def edit_scatter(self):
        if self.fileDir != None:
            self.edit_output, self.edit_plot = plt.subplots(figsize=(12, 10))

            plt.imshow(self.input[self.current_num], cmap='gray')

            self.xdata = []
            self.ydata = []

            for y, x in self.pred[self.current_num]:
                self.scat_list.append(self.edit_plot.scatter(x, y, s=20, marker='.', c='b'))

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Interactive Plot')

            self.edit_plot.set_aspect('auto', adjustable='box')

            self.line, = self.edit_plot.plot(self.xdata, self.ydata)

            cid = plt.connect('button_press_event', self.add_point)
            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.about(self, 'Notice', 'please Load IMG.')

    def add_point(self, event):
        if event.inaxes != self.edit_plot:
            return

        if event.button == 1:
            self.scat_list.append(self.edit_plot.scatter(event.xdata, event.ydata, s=20, marker='.', c='b'))
            plt.draw()

        if event.button == 3:
            remove_x = event.xdata
            remove_y = event.ydata

            self.scat_list[-1].remove()
            self.scat_list.pop()

            plt.draw()

        if event.button == 2:
            self.update_canvas()
            plt.close(self.edit_output)

    def update_canvas(self):
        print(len(self.scat_list))
        if len(self.scat_list) != 0 and len(self.text_list) != 0:
            for scat, text in zip(self.scat_list, self.text_list):
                scat.remove()
                text.remove()

            self.scat_list = []
            self.text_list = []

            self.output_canvas.draw()

    def list_click(self):
        if self.fileDir != None:
            self.current_num = self.file_list.currentRow()

            self.vis_output.clear()
            self.vis_origin.clear()

            self.pred_plot = self.vis_output.add_subplot(111)
            self.origin_plot = self.vis_origin.add_subplot(111)

            self.origin_plot.axis('off')
            self.pred_plot.axis('off')

            self.pred_plot.imshow(self.input[self.current_num], cmap='gray')
            self.origin_plot.imshow(self.input[self.current_num], cmap='gray')

            for idx, i in enumerate(self.pred[self.current_num]):
                self.pred_plot.scatter(int(i[1]), int(i[0]), s=20, marker='.', c='b')
                self.pred_plot.text(i[1] + 0.02, i[0] + 0.02, f'{idx + 1}', c='r', fontsize=7)

            self.output_canvas.draw()
            self.origin_canvas.draw()

    def Delete_list(self):

        # if self.current_num != -1 and self.current_num != None and self.file_list.count() != 0:
        del_num = self.file_list.currentRow()
        if del_num != -1:
            self.file_list.takeItem(del_num)

            self.input.pop(del_num)
            self.pred.pop(del_num)

        else:
            pass
        # 현재 선택된 표시된 이미지 출력했을 때 다음 이미지로 뜨게 처리

    def excel_save(self):
        self.savedir = QFileDialog.getExistingDirectory(self, "Open Directory", r'./')
        write_wb = openpyxl.Workbook()

        write_ws = write_wb.create_sheet('Sheet1')
        write_ws = write_wb.active
        # for i in range(1, 38):
        #     write_ws[f'{str(i)}']
        # write_ws['B1'] = 'Sella'
        # write_ws['A2'] = self.file_list.currentItem().text()
        # write_ws['B2'] = 0

        # for line, name in zip(range(66, 91), self.landmark_name):
        #     write_ws[f'{line}1'] = name

        lll = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM']

        for idx, num in enumerate(lll):
            write_ws[f'{num}1'] = idx + 1

        for idx in range(self.file_list.count()):
            write_ws[f'A{idx + 2}'] = self.file_list.item(idx).text()
            for idx2, i in enumerate(lll):
                pos = self.pred[idx][idx2].detach().cpu().numpy()
                write_ws[f'{i}{idx+2}'] = f'{int(pos[0])}, {int(pos[1])}'

        write_wb.save('./test.xlsx')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
