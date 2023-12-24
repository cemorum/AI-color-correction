import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from unet_main import load_model
from shadows import predict_shadows

def predict_and_save(input_image_path, output_image_path, model, flares=False, shadows=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    im = Image.open(input_image_path).convert('RGB')
    original_size = im.size
    width = (original_size[0] // 256 + 1) * 256
    height = (original_size[1] // 256 + 1) * 256

    new_size = (width, height)

    im = im.resize(new_size, Image.LANCZOS)

    # Примените преобразование к изображению и преобразуйте его в float32
    tensor_image = transform(im).to(dtype=torch.float32)

    # Измените размерность тензора, чтобы соответствовать ожидаемому входу модели
    # Например, изменение размера до (1, C, H, W)
    # Предполагая, что ваша модель ожидает четырехмерный тензор (batch_size, channels, height, width)
    tensor_image = tensor_image.unsqueeze(0)  # Добавление размерности batch

    # Отправьте тензор на устройство (например, GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_image = tensor_image.to(device)

    with torch.no_grad():
        prediction = model(tensor_image)

    if not flares:
        prediction[prediction < 0] *= -1
        prediction[prediction > 1] = 1


    # Преобразование обратно в PIL Image
    output_image = transforms.ToPILImage()(prediction.squeeze().cpu().detach())
    output_image = output_image.resize(original_size, Image.LANCZOS)
    output_image.save(output_image_path)

def predict_image(input_file, output_file, model_path, flares=False, shadows=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = load_model(model_path, n_channels=3, n_classes=3)
    net.to(device)

    if shadows:
        predict_shadows(output_file, output_file, net)
    else:
        predict_and_save(input_file, output_file, net, flares, shadows)
