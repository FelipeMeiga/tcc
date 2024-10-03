import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os

# Função para mostrar uma imagem
def imshow(img):
    img = img / 2 + 0.5  # dessormaliza
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Transformações para os dados de treinamento e teste
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregando o dataset MNIST
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

# Função para mostrar um lote de imagens de treinamento
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Mostrar imagens
imshow(utils.make_grid(images))

# Definindo a arquitetura da CNN para MNIST
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 1 canal de entrada para MNIST
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Ajuste para 28x28 imagens após 3 conv/pool camadas
        self.fc2 = nn.Linear(512, 10)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 3 * 3)  # Ajuste para 28x28 imagens após 3 conv/pool camadas
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    def generate_images(self, model, device, dataloader, num_images=5):
        model.eval()
        dataiter = iter(dataloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Mostrar algumas imagens geradas pela CNN
        imshow(utils.make_grid(images[:num_images].cpu()))
        print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(num_images)))


# Inicializando o modelo, função de perda e otimizador
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função de treinamento
def train(model, device, trainloader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}')

# Função de teste
def test(model, device, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(testloader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100. * correct / len(testloader.dataset):.0f}%)\n')

"""
# Treinando e testando o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 11):
    train(model, device, trainloader, optimizer, criterion, epoch)
    test(model, device, testloader, criterion)
"""

model_path = './cnn/cnn_mnist.pth'

"""
# Salvando os pesos do modelo
torch.save(model.state_dict(), model_path)
print(f'Model weights saved to {model_path}')
"""

# Função para gerar imagens com a CNN treinada

"""
# Carregar os pesos do modelo salvo
loaded_model = CNN()
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)

# Gerar imagens com o modelo carregado
generate_images(loaded_model, device, testloader)
"""
