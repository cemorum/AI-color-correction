import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from unet_main import *

def predict_shadows(input_file, output_file, attention_unet):
    print('----', input_file, output_file)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # attention_unet = AttentionUNet(n_classes=3).to(device)
    # attention_unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # attention_unet.eval()

    with torch.no_grad():
        transform = transforms.ToTensor()

        im = Image.open(input_file).convert('RGB')
        original_size = im.size
        tensor_image = transform(im).to(dtype=torch.float32)

        tensor_image = tensor_image.unsqueeze(0)
        required_size = (1, 3, 256, 256)
        if tensor_image.size() != required_size:
            tensor_image = torch.nn.functional.interpolate(tensor_image, size=(required_size[2], required_size[3]), mode='bilinear', align_corners=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor_image = tensor_image.to(device)

        prediction = attention_unet(tensor_image)
        prediction[prediction < 0 ] *= -1
        prediction[prediction > 1 ] = 1

        prediction_resized = torch.nn.functional.interpolate(prediction, size=original_size[::-1], mode='bilinear', align_corners=False)

        output_image = transforms.ToPILImage()(prediction_resized.squeeze().cpu().detach())

    mask_image = cv2.imread(input_file)

    target_image = np.array(output_image)

    target_image[:, :, 1] = 0
    target_image[:, :, 2] = 0

    masked_image = cv2.addWeighted(target_image, 1, mask_image, 1, 1)

    output_image_path = output_file
    cv2.imwrite(output_image_path, masked_image)


# predict_shadows(r'C:\Users\xgorio\Desktop\OrangeHood_WEB\shadows_city.jpg', 
#                 r'C:\Users\xgorio\Desktop\OrangeHood_WEB\masked_image.jpg',
#                 r'C:\Users\xgorio\Desktop\OrangeHood_WEB\shad_weights_0.0079.pth')