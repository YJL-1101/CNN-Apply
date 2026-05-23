from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import torch
from model import GoogLeNet,Inception
from PIL import Image



#数据处理
def image_process(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.486, 0.453, 0.415), (0.29731594,0.35303732,0.38785615))
    ])


    image = transform(image)

    image = image.unsqueeze(0)
    image = image.to(device)

    return image

#模型推理
def model_inference(model,image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型放入设备中
    model = model.to(device)
    model.eval()

    classes = ['Apple',
               'Banana',
               'Cherry',
               'Grape',
               'Lemon',
               'Orange',
               'Peach',
               'Pear',
               'Raspberry',
               'Strawberry',
               'Watermelon'
               ]

    with torch.no_grad():
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
        print("预测值：",classes[result])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoogLeNet(Inception)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    image = image_process('./img_1.png')

    model_inference(model,image)