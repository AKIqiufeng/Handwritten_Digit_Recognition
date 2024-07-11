import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.net import *
from random import randint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model.load_state_dict(torch.load('./LeNet/mnist_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict(image_path):
    print(image_path)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Prediction" + str(int(predicted.item())))
    plt.show()
    return predicted.item()


for i in range(3):
    test_n = randint(0, 9)
    idx = randint(1, 10)
    result = predict(f"./number_img/{str(test_n)}/{str(test_n)}_{str(idx)}.jpg")
    print(result)