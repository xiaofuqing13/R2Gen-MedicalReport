import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
import torch
from torchvision import transforms
from PIL import Image

# 假设你已经加载了模型
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('model_iu_xray.pth', map_location=torch.device('cpu')))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("医学图像分析工具")
        self.setGeometry(100, 100, 600, 400)

        # 创建界面组件
        self.label = QLabel("请选择一张医学图像进行分析", self)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(200, 200)
        self.result_label = QLabel("预测结果：", self)

        self.upload_button = QPushButton("上传图像", self)
        self.upload_button.clicked.connect(self.upload_image)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.upload_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            # 显示图像
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(200, 200))

            # 推理
            image = Image.open(file_path).convert('L')
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()

            # 显示结果
            self.result_label.setText(f"预测结果：{prediction}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())