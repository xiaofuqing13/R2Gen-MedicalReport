import gc
import sys
import json
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image
import torch
from torchvision import transforms
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QThreadPool, QObject



def handle_exception(exc_type, exc_value, exc_traceback):
    # 输出错误信息和堆栈跟踪
    print("未处理的异常：", exc_type, exc_value)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

# 导入模型
sys.path.append('./models')
sys.path.append('./modules')

from models.r2gen import R2GenModel
from modules.tokenizers import Tokenizer

# 设置参数
args = argparse.Namespace()
args.max_seq_length = 60
args.max_seq_num = 3
args.visual_extractor = 'resnet101'
args.visual_extractor_pretrained = False
args.d_model = 512
args.d_ff = 2048
args.d_vf = 2048
args.drop_prob_lm = 0.5
args.num_heads = 8
args.num_layers = 3
args.exp_name = 'r2gen'
args.device = 'cpu'
args.ann_path = 'data/iu_xray/processed_annotation.json'
args.threshold = 3
args.dataset_name = 'iu_xray'
args.bos_idx = 0
args.eos_idx = 1
args.pad_idx = 2
args.use_bn = True  # 使用批量归一化，通常设置为 True
args.dropout = 0.5
args.rm_num_slots = 3
args.rm_num_heads = 8
args.rm_d_model = 512

# 初始化 tokenizer 和模型
tokenizer = Tokenizer(args)
args.vocab_size = len(tokenizer.token2idx)  # 确保正确获取词汇表大小

# 初始化模型
model = R2GenModel(args, tokenizer)

# 加载模型权重
model_weights = torch.load('model_iu_xray.pth', map_location='cpu')
model_state_dict = model.state_dict()
model.eval()
torch.set_num_threads(4)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(224),          # 先缩放到 256x256
    transforms.CenterCrop(224),      # 中心裁剪为 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 线程互斥锁，用于避免线程冲突
mutex = QMutex()


class InferenceWorker(QObject):
    finished = pyqtSignal(str)

    def __init__(self, model, transform, tokenizer):
        super().__init__()
        self.model = model
        self.transform = transform
        self.tokenizer = tokenizer

    def process(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                output = self.model.sample(input_tensor, sample_method='greedy')[0]
                report = self.tokenizer.decode(output, skip_special_tokens=True)
                self.finished.emit(report)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")


class InferenceThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, parent=None, image=None, model=None, transform=None, tokenizer=None):
        super().__init__(parent)
        self.image = image  # 直接传入已加载的 PIL.Image 对象
        self.model = model
        self.transform = transform
        self.tokenizer = tokenizer

    def run(self):
        mutex.lock()
        try:
            # 直接使用 self.image，无需重新加载
            input_single = self.transform(self.image).unsqueeze(0)  # [1,3,H,W]

            # 模拟双视图输入（适配模型）
            input_dual = torch.cat([input_single, input_single], dim=0)  # [2,3,H,W]
            input_dual = input_dual.unsqueeze(0)  # [1,2,3,H,W]

            # 生成报告
            with torch.no_grad():
                output = self.model(input_dual, mode='sample')

            # 解码文本
            report = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.result_signal.emit(report)
        except Exception as e:
            self.result_signal.emit(f"错误: {str(e)}")
        finally:
            mutex.unlock()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("医学图像分析工具")
        self.setGeometry(100, 100, 600, 400)

        self.label = QLabel("请选择一张医学图像进行分析", self)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(224, 224)
        self.result_label = QLabel("生成的报告：", self)

        self.upload_button = QPushButton("上传图像", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.generate_button = QPushButton("生成报告", self)
        self.generate_button.clicked.connect(self.generate_report)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.generate_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 初始化线程池
        self.thread_pool = QThreadPool()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(224, 224))
            self.image_path = file_path  # 保存图像路径

    def generate_report(self):
        if hasattr(self, 'image_path'):
            # 加载图像并转换为 PIL.Image 对象
            image = Image.open(self.image_path).convert("RGB")

            # 传递 image 对象而非路径
            inference_thread = InferenceThread(
                parent=self,
                image=image,  # 关键修改：传递已加载的图像
                model=model,
                transform=transform,
                tokenizer=tokenizer  # 传递 tokenizer
            )
            inference_thread.result_signal.connect(self.update_result)
            inference_thread.start()

    def update_result(self, output):
        """ 更新 UI 显示报告 """
        print(f"报告输出：{output}")
        self.result_label.setText(f"生成的报告：{output}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
