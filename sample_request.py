
import requests
import base64
import cv2 
from PIL import Image
from io import BytesIO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='additional_input/Hanekawa_wild.png', help='Input image or folder')
parser.add_argument('--tile', type=int, default=0, help='Input image or folder')


def input_fn(img_str):
    """
    Deserialize and prepare the prediction input
    """

    data = base64.b64decode(img_str, validate= True)

    im = Image.open(BytesIO(data))
    # np_image = np.array(im)
    # im.save('image1.png', 'PNG')

    return im



def prepare_image(frame, encode_quality = 100): 
  frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
  _,buffer = cv2.imencode('.png', frame, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
  dashboard_img = base64.b64encode(buffer).decode()
  return dashboard_img



def envoke_endpoint(image ,tile=512, half = False, output_dir="results"): 
      

  # image = cv2.imread(img_name)

  img_str = prepare_image(image)

  post_dict= {
    "img_str": img_str, 
    "tile": tile,
    "half": half,
    "output_dir" :  output_dir
  }


  result = requests.post("http://127.0.0.1:8000/inference", json=post_dict)


  output_image = input_fn(result.json())

  # output_image.save(img_name.split(".")[0]+"_out.png")

  return output_image
