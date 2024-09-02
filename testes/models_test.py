import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from ignite.metrics import FID
from ignite.engine import Engine
from vae_trace import VAE
from cnn_mnist import CNN

# Configuração
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dims_color = 28*28

# Transformações e carregamento do dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Função para converter imagens de 1 canal para 3 canais
def convert_to_rgb(images):
    return images.repeat(1, 3, 1, 1)

# Função para redimensionar imagens para 299x299
resize = transforms.Resize((299, 299), antialias=True)

# Carregar o modelo VAE
vae_model = VAE(image_dims_color).to(device)
vae_model.load_state_dict(torch.load('./epochs/vae_epoch_15.pth'))

# Carregar o modelo CNN
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load('./cnn/cnn_mnist.pth'))

# Função para reconstruir imagem com VAE
def reconstruct_with_vae(model, image):
    model.eval()
    with torch.no_grad():
        image = image.view(-1, image_dims_color).to(device)
        recon_image, _, _ = model(image)
        recon_image = recon_image.view(1, 1, 28, 28).cpu()
    return recon_image

# Função para reconstruir imagem com CNN
def reconstruct_with_cnn(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        recon_image = model(image)
        recon_image = recon_image.argmax(dim=1, keepdim=True).cpu().float()
    return recon_image

# Função de predição para o avaliador
def process_function(engine, batch):
    y_pred, y_true = batch
    return y_pred, y_true

# Define o avaliador
evaluator = Engine(process_function)

# Configura a métrica FID
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
inception_model.aux_logits = False  # Desativa as camadas auxiliares
inception_model.eval()  # Coloca o modelo em modo de avaliação
inception_model.to(device)

fid_metric = FID(num_features=2048, feature_extractor=inception_model)
fid_metric.attach(evaluator, "fid")

# Selecionar uma imagem aleatória do conjunto de dados real
for real_image, _ in dataloader:
    break  # Pegar apenas a primeira imagem do iterador

# Reconstruir a imagem com VAE
vae_recon_image = reconstruct_with_vae(vae_model, real_image)

# Reconstruir a imagem com CNN
cnn_recon_image = reconstruct_with_cnn(cnn_model, real_image)

# Converter para RGB e redimensionar para 299x299
real_image_rgb = convert_to_rgb(real_image)
vae_recon_image_rgb = convert_to_rgb(vae_recon_image)
cnn_recon_image_rgb = convert_to_rgb(cnn_recon_image)

real_image_rgb = resize(real_image_rgb)
vae_recon_image_rgb = resize(vae_recon_image_rgb)
cnn_recon_image_rgb = resize(cnn_recon_image_rgb)

# Calcular FID para VAE
state = evaluator.run([[vae_recon_image_rgb, real_image_rgb]])
fid_vae = state.metrics["fid"]

# Resetar o FID metric para calcular para o CNN
fid_metric.reset()

# Calcular FID para CNN
state = evaluator.run([[cnn_recon_image_rgb, real_image_rgb]])
fid_cnn = state.metrics["fid"]

print("FID VAE:", fid_vae)
print("FID CNN:", fid_cnn)
