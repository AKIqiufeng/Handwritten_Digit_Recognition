import torch
import torchvision.transforms as transforms
from PIL import Image
from models.net import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model.load_state_dict(torch.load('./LeNet/epoch/module_30.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()


def evaluate_folder(folder_path):
    total_images = 0
    correct_predictions = 0
    results = []

    for label in range(10):
        label_folder = os.path.join(folder_path, str(label))
        if not os.path.isdir(label_folder):
            continue

        image_files = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]

        for image_file in image_files:
            image_path = label_folder + "/" + image_file

            predicted_label = predict(image_path)

            print("预测值:", predicted_label, "实际值：", label)

            if predicted_label == label:
                correct_predictions += 1
            total_images += 1

            results.append(f'{image_path}\t{label}\t{predicted_label}\n')

    accuracy = (correct_predictions / total_images) * 100

    with open('prediction_results.txt', 'w') as f:
        f.writelines(results)

    return accuracy


folder_path = './number_img/'
accuracy = evaluate_folder(folder_path)

print(f'总准确率: {accuracy:.2f}%')
