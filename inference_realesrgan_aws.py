import argparse
import cv2
import glob
import os
from torchvision.transforms import functional
import sys

sys.modules["torchvision.transforms.functional_tensor"] = functional
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def input_fn(img_str):
    """
    Deserialize and prepare the prediction input
    """

    data = base64.b64decode(img_str, validate= True)

    im = Image.open(BytesIO(data))
    np_image = np.array(im)
    # im.save('image1.png', 'PNG')

    return np_image


def prepare_image(frame, encode_quality = 50): 
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    _,buffer = cv2.imencode('.png', frame, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
    dashboard_img = base64.b64encode(buffer).decode()
    return dashboard_img


def main(img_str , model_name = "RealESRGAN_x4plus", outscale=4, tile=0, tile_pad=10,
          pre_pad=0, half=False, ext= 'auto', gpu_id= None, output_dir="results", suffix= "out"):

    """Inference demo for Real-ESRGAN.
    """

    # determine models according to model names
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']


    # determine model paths
    if os.path.exists(os.path.join('weights', model_name + '.pth')):
        model_path = os.path.join('weights', model_name + '.pth')
    else:
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=gpu_id)


    os.makedirs(output_dir, exist_ok=True)




    img = input_fn(img_str=img_str)

    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        save_path= os.path.join(output_dir,"output.png")
        cv2.imwrite(save_path, output)
    
    return prepare_image(output)


