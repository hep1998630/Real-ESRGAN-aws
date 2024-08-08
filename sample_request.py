
import requests
import base64
import cv2 
from PIL import Image
import numpy as np
from io import BytesIO

def input_fn(img_str):
    """
    Deserialize and prepare the prediction input
    """

    data = base64.b64decode(img_str, validate= True)

    im = Image.open(BytesIO(data))
    # np_image = np.array(im)
    # im.save('image1.png', 'PNG')

    return im



def prepare_image(frame, encode_quality = 50): 
    _,buffer = cv2.imencode('.png', frame, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
    dashboard_img = base64.b64encode(buffer).decode()
    return dashboard_img




image = cv2.imread("/media/mohamed/4tb/work/aws_nd/capstone/dev2k_dataset/DIV2K_valid_LR_x8/0801x8.png")

img_str = prepare_image(image)

post_dict= {
  "img_str": img_str, 
  "half": False,
  "output_dir" :  "results"
}


result = requests.post("http://127.0.0.1:8000/inference", json=post_dict)


output_image = input_fn(result.json())

output_image.save("output.png")
