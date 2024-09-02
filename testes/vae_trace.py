import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random

transform = transforms.Compose([
    transforms.ToTensor()
])

image_dims_color = 28*28

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
#torchvision.datasets.WIDERFace

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class VAE(nn.Module):
    def __init__(self, image_dims_color):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_dims_color, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc51 = nn.Linear(64, 20)  # mean
        self.fc52 = nn.Linear(64, 20)  # log variance
        self.fc6 = nn.Linear(20, 64)
        self.fc7 = nn.Linear(64, 128)
        self.fc8 = nn.Linear(128, 256)
        self.fc9 = nn.Linear(256, 512)
        self.fc10 = nn.Linear(512, image_dims_color)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))
        h4 = torch.relu(self.fc4(h3))
        return self.fc51(h4), self.fc52(h4)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h6 = torch.relu(self.fc6(z))
        h7 = torch.relu(self.fc7(h6))
        h8 = torch.relu(self.fc8(h7))
        h9 = torch.relu(self.fc9(h8))
        return torch.sigmoid(self.fc10(h9))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, image_dims_color))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, image_dims_color), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = VAE(image_dims_color).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

specific_image, _ = train_dataset[0]
specific_image = specific_image.to(device).view(-1, image_dims_color)

model.train()
num_epochs = 15
reconstructions = []

"""
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    # Salvar os pesos do modelo para cada época
    torch.save(model.state_dict(), f'./epochs/vae_epoch_{epoch + 1}.pth')
    
    # Reconstruir a imagem específica
    model.eval()
    with torch.no_grad():
        recon_image, _, _ = model(specific_image)
        reconstructions.append(recon_image.view(28, 28).cpu().numpy())
    model.train()
    
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
"""

def predict(image, num_epochs):
    image = image.to(device).view(-1, image_dims_color)
    predictions = []
    for epoch in range(1, num_epochs + 1):
        model.load_state_dict(torch.load(f'./epochs/vae_epoch_{epoch}.pth'))
        model.eval()
        with torch.no_grad():
            recon_image, _, _ = model(image)
            predictions.append(recon_image.view(28, 28).cpu().numpy())
    return predictions

model.eval()
with torch.no_grad():
    z = torch.randn(64, 20).to(device)
    sample = model.decode(z).cpu()
    sample = sample.view(64, 28, 28)

specific_image, _ = test_dataset[random.choice(range(0, test_size))]
predictions = predict(specific_image, num_epochs)

fig, ax = plt.subplots(4, 4, figsize=(12, 12))
ax = ax.flatten()
ax[0].imshow(specific_image.view(28, 28).cpu(), cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

for i in range(1, 16):
    ax[i].imshow(predictions[i-1], cmap='gray')
    ax[i].set_title(f'Epoch {i}')
    ax[i].axis('off')

plt.show()
