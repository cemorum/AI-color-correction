from torchvision import transforms
from PIL import Image
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os
from unet_main import load_model

def predict(input_frame, model, frame_number, save_folder, flares=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Convert the NumPy frame to a PIL Image
    input_frame = Image.fromarray(input_frame)

    original_size = input_frame.size

    # Ensure dimensions are multiples of 256
    width = (original_size[0] // 256 + 1) * 256
    height = (original_size[1] // 256 + 1) * 256

    # Resize the image
    im = input_frame.resize((width, height), Image.LANCZOS)

    # Convert the image to a tensor
    im_tensor = transform(im)

    # Process with the model
    # Note: You'll need to adjust the following line to fit your model's requirements
    output = model(im_tensor.unsqueeze(0))

    # Save the processed frame
    save_path = os.path.join(save_folder, f'frame_{frame_number}.png')
    output_image = transforms.ToPILImage()(output.squeeze().cpu().detach())
    output_image.save(save_path)


def predict_video(video_path, save_folder, model_path ):
    net = load_model(model_path, 3, 3)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predict(rgb_frame, net, frame_number, save_folder)
        frame_number += 1

    cap.release()