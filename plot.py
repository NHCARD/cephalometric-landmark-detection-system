import os
import sys

import cv2
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from torch.utils.data import DataLoader

from utils.dataload import *
from utils.utils import *
import openpyxl


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.H = 800
        self.W = 640
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = load_model('predict')
        self.pred = []
        self.input = []
        self.fileDir = None
        self.current_num = 0

        self.initUI()
        self.setLayout(self.layout)
        self.setWindowTitle('Landmark Detection System')
        self.setGeometry(200, 200, 1400, 800)  # 창의 위치 x좌표, y좌표, 가로크기, 세로 크기
        self.setFixedSize(1400, 800)

    def initUI(self):
        self.vis_output = plt.Figure(figsize=(10, 10))  # 모델 아웃풋 이미지
        self.vis_origin = plt.Figure(figsize=(5, 5))  # 모델 인풋 이미지
        self.origin_canvas = FigureCanvas(self.vis_origin)  # 인풋 이미지 출력 박스
        self.output_canvas = FigureCanvas(self.vis_output)  # 아웃풋 이미지 출력 박스

        menu_load_img = self.create_action('&Load Image', 'Ctrl+l', 'Load to open Image', self.img_load)
        menu_load_dir = self.create_action('&Load Folder', 'Ctrl+d', 'Load to Image Folder', self.load_directory)
        menu_edit = self.create_action('&Edit', 'Ctrl+e', 'Image Edit to Scatter', self.edit_scatter)
        menu_excel = self.create_action('&Save Excel', 'Ctrl+x', 'Save Excel File', self.excel_save)

        menubar = QMenuBar(self)  # 메뉴바
        file_menu = menubar.addMenu('&File')
        file_menu.addActions([menu_load_img, menu_load_dir, menu_edit, menu_excel])

        layout = QHBoxLayout()  # 메인 레이아웃 (Horizon)
        layout.addWidget(self.origin_canvas)
        layout.addWidget(self.output_canvas)

        btn_layout = QVBoxLayout()  # 버튼, 리스트 레이아웃
        directory_btn = self.create_button("Load Directory", self.load_directory)
        load_img_btn = self.create_button("Load Image", self.img_load)
        edit_btn = self.create_button("Edit Landmarks", self.edit_scatter)
        self.del_list = self.create_button("Delete", self.delete)

        self.file_list = QListWidget(self)
        self.file_list.itemDoubleClicked.connect(self.list_click)

        btn_layout.addWidget(load_img_btn)
        btn_layout.addWidget(directory_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(self.file_list)
        btn_layout.addWidget(self.del_list)

        layout.addLayout(btn_layout)
        layout.setMenuBar(menubar)

        self.layout = layout

    def create_action(self, text, shortcut, status_tip, slot):
        action = QAction(text, self)
        action.setShortcut(shortcut)
        action.setStatusTip(status_tip)
        action.triggered.connect(slot)
        return action

    def create_button(self, text, slot):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        return btn

    def img_load(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(self, "Open Image", r'./CL-Data/dataset/val/0img',
                                                      self.tr("Video Files (*.png)"))

        if self.fileDir != '':
            img = cv2.imread(self.fileDir)
            self.orig_H, self.orig_W = img.shape[:2]

            find_idx = self.fileDir.rindex('/')
            self.file_list.addItem(self.fileDir[find_idx + 1:])

            self.test_data = DataLoader(dataload_valid(path=self.fileDir, H=self.H, W=self.W, aug=False, mode='img'))
            self.predict()
        else:
            pass

        self.current_num = self.file_list.count() - 1

    def load_directory(self):
        self.fileDir = QFileDialog.getExistingDirectory(self, "Open Folder", r'./CL-Data/dataset/val')

        if self.fileDir != '':
            self.test_data = DataLoader(dataload_valid(path=self.fileDir, H=self.H, W=self.W, aug=False, mode='dir'),
                                        batch_size=1, shuffle=False, num_workers=5)

            f_list = os.listdir(self.fileDir)

            for i in f_list:
                self.file_list.addItem(f'{i}')

            self.current_num = self.file_list.count() - 1

            img = cv2.imread(glob.glob(self.fileDir + '/*.png')[0])

            self.orig_H = img.shape[0]
            self.orig_W = img.shape[1]

            self.predict()
        else:
            pass

    def predict(self):
        Ymap, Xmap = np.mgrid[0:self.H:1, 0:self.W:1]
        Ymap, Xmap = torch.tensor(Ymap.flatten(), dtype=torch.float).unsqueeze(1).to(self.device), \
            torch.tensor(Xmap.flatten(), dtype=torch.float).unsqueeze(1).to(self.device)

        with torch.no_grad():
            for inputs in self.test_data:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                pred = torch.cat([argsoftmax(outputs[0].view(-1, self.H * self.W), Ymap, beta=1e-3) * (self.orig_H / self.H),
                                  argsoftmax(outputs[0].view(-1, self.H * self.W), Xmap, beta=1e-3) * (self.orig_W / self.W)],
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

            self.xdata, self.ydata = [], []
            for y, x in self.pred[self.current_num]:
                self.scat_list.append(self.edit_plot.scatter(x, y, s=20, marker='.', c='b'))

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Interactive Plot')

            self.edit_plot.set_aspect('auto', adjustable='box')

            self.line, = self.edit_plot.plot(self.xdata, self.ydata)

            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.about(self, 'Notice', 'Please Load Image.')

    def add_point(self, event):
        if event.inaxes != self.edit_plot:
            return

        if event.button == 1:
            self.scat_list.append(self.edit_plot.scatter(event.xdata, event.ydata, s=20, marker='.', c='b'))
            plt.draw()

        if event.button == 2:
            self.update_canvas()
            plt.close(self.edit_output)

        if event.button == 3:
            self.scat_list[-1].remove()
            self.scat_list.pop()
            plt.draw()

    def update_canvas(self):
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

    def delete(self):
        del_num = self.file_list.currentRow()
        if del_num != -1:
            self.file_list.takeItem(del_num)
            self.input.pop(del_num)
            self.pred.pop(del_num)

        else:
            pass
        # 현재 선택된 표시된 이미지 출력했을 때 다음 이미지로 뜨게 처리

    def excel_save(self):
        self.savedir = QFileDialog.getSaveFileName(self, "Open Directory", r'./', 'xlsx files (*.xlsx)')
        write_wb = openpyxl.Workbook()

        write_ws = write_wb.create_sheet('Sheet1')
        write_ws = write_wb.active

        lll = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM']

        for idx, num in enumerate(lll):
            write_ws[f'{num}1'] = idx + 1

        for idx in range(self.file_list.count()):
            write_ws[f'A{idx + 2}'] = self.file_list.item(idx).text()
            for idx2, i in enumerate(lll):
                pos = self.pred[idx][idx2].detach().cpu().numpy()
                write_ws[f'{i}{idx+2}'] = f'{int(pos[0])}, {int(pos[1])}'

        write_wb.save(self.savedir[0])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
