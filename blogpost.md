# Deployment of Real-ESRGAN in AWS using Docker, FastAPI, Uvicorn, and Gradio

## Definition 
### Project Overview
This blog post is a walkthrough in deploying [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) in Amazon Web Services (AWS). Real-ESRGAN is a state-of-the-art GAN-based model for image super resolution tasks. Real-ESRGAN improves opon its ancestor the [ESRGAN](https://github.com/xinntao/ESRGAN) in the means of inference time, making it a suitable option for real-time image super-resoltuion tasks. In this blog post, I will explain how you can deploy Real-ESRGAN in AWS and build a web application for your users to interact with it.   
### Utilized Techologies  
This project utilizes the following technologies to deploy REal-ESRGAN on AWS: 
- **Real-ESRGAN** : The GAN-based model by [@Xinntao](https://github.com/xinntao) which provides real-time inference capabilities for image super-resolution.  
- **Amazon EC2 instance** : This is the main AWS service I am going to use. Amazon Elastic Compute Cloud (Amazon EC2) provides on-demand, scalable computing capacity in the Amazon Web Services (AWS) Cloud as explained in the [AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html). 
- **Docker** : I used [Docker](https://www.docker.com/) as a containarization framework to easily deploy and launch my application in AWS. 
- **FastAPI** : [FastAPI](https://fastapi.tiangolo.com/) is used as a web-framework. FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints [docs](https://fastapi.tiangolo.com/). 
- **Uvicorn** : [Uvicorn](https://www.uvicorn.org/) is used as an ASGI web server. 
- **Gradio** : Gradio is used to build user-interface to interact with the application. 

### Model Information 
Real-ESRGAN aims at developing Practical Algorithms for General Image/Video Restoration. It extends the powerful ESRGAN to a practical restoration application, namely, Real-ESRGAN. Real-ESRGAN can deal with various degredation effects like unknown blurs, complicated noises and common compression artifacts. The real degredations usually come from complicated combinations of different degredation processes such as the imaging system of cameras, image editing, and internet transmission. This motivated the authors to extend the classical first-order degredation model to high-order degredation modeling for real-world degredation. As for the network archeticture, Real-ESRGAN adopts the same generator network as that in ESRGAN. For the scale factor of ×2 and ×1, it first employs a pixel-unshuffle operation to reduce spatial size and re-arrange information to the channel dimension. 
<p align="center">
  <img src="assets/Real-ESRGAN Arch.png">
</p>

### Model Results
Below are some qualitative results of Real-ESRGAN
<p align="center">
  <img src="assets/teaser.jpg">
</p>


## Methodology
### System Archeticture


### Adapt Real-ESRGAN code to accept http requests  
The inference code of Real-ESRGAN needs to be adapted to accept http requests from the web application. To achieve this, I used FastAPI as a web framework and Uvicorn as an ASGI server. The main modifications are the following: 

- Call the FastAPI class and instantiate an app from it 

```python
from fastapi import FastAPI
app = FastAPI()
```
- Use the app instance to define endpoints for your application, note the request type and the endpoint path. 

```python
@app.get("/")
def read_root():
	return {"Hello": "World"}

@app.post("/inference")
def inference(item: data):
	result = main(item.img_str, tile=item.tile)
	
	return result

```
- To pass arguemnts to one of your endpoints, you can use pydantic to define a class that inf=herets from the BaseModel class. Any method you define for this class can be passed in the request body as json key-value pairs

```python
from pydantic import BaseModel

class data(BaseModel):
	img_str: str
	tile: int = 0
	half: bool = False
	output_dir: str = "results"
```



Here is the full main.py script: 

```python
from fastapi import FastAPI
from inference_realesrgan_aws import main
from pydantic import BaseModel


class data(BaseModel):
	img_str: str
	tile: int = 0
	half: bool = False
	output_dir: str = "results"



app = FastAPI()

@app.get("/")
def read_root():
	return {"Hello": "World"}



@app.post("/inference")
def inference(item: data):
	result = main(item.img_str, tile=item.tile)
	
	return result
```

Another point you need to consider is how to send and recieve data. There are several different ways to do this, but I used simple base64 encoding for the image to send it as a string. The functions below are from inference_realesrgan_aws.py script to encode and decode the data

```python
def input_fn(img_str):
    """
    Deserialize and prepare the prediction input
    """

    data = base64.b64decode(img_str, validate= True)

    im = Image.open(BytesIO(data))
    np_image = np.array(im)
    # im.save('image1.png', 'PNG') # Optionally save the image 

    return np_image


def prepare_image(frame, encode_quality = 50): 
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    _,buffer = cv2.imencode('.png', frame, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
    dashboard_img = base64.b64encode(buffer).decode()
    return dashboard_img
```



You need to update the main function in your Real-ESRGAN inference script to accept your arguments. This is trivial, so I am not going to discuss it here, but the code is available in inference_realesrgan_aws.py script. 


### Create a sample function to send http requests 
You need to create a scirpt to interact with your server using http requests. The script should send low-resolution images with proper arguments and recieve the upscaled image. The code uses the requests library and applies the same encoding and decoding methods in the inference script we discussed earlier, it is available in sample_request.py 

### Create a web demo using Gradio 
The web UI of the application is created using Gradio. Gradio allows you to quickly build a demo or web application for your machine learning model. The script is available in gradio_demo.py 


### Create a Dcoker container for easier deployment 
To easily deploy my application in AWS, I created a Docker container. The Dockerfile used to build the container will take care of installing all the needed dependencies. You can take a look at the Dockerfile for more details. To build the container, use the following command in the project directory. 
```bash
docker build -t real_esrgan .  
```
You can optionally push the container to your repository in [Docker Hub](https://hub.docker.com/) to pull it later in AWS. 

### Run EC2 instance with GPU capabilities
Now, we finally get to work on AWS infrastructure! We will launch an EC2 instance to host both the Real-ESRGAN ASGI server and the gradio demo app. Start by navigaring to the EC2 dashboard in your AWS account 
<p align="center">
  <img src="assets/EC2_dashboard.png">
</p>

Then select **Launch Inastance** from the top right corner, this will open a page to configure settings for your instance. 

#### AMazon Machine Image (AMI) and instance type
Select an AMI that supports deep learning applications, to make our life easier as it will have all the required software to build deep learning applications. Moreover, you need to select an instanced type that supports accelerated computing using Nvidia GPUs. This way, we can leverage the parallel computing of GPUs wsint pytorch with cuda which will significantly afect the latency of our application. One example of n AMI and instance type is shown below. 
<p align="center">
  <img src="assets/EC2_AMI.png">
</p>

#### Key pair and netwrok settings
Create a key pair and save it in a safe location. This key pair will be used to connect to your EC2 instance using ssh or putty. Instructions on how to connect are explained the AWS [docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html).   
For network settings, it is better to disable http requests as you will not need them for this app (since gradio will take care of creating a proxy server for you). As for ssh, make sure to allow the IP of the computer you will use to access the server. 

You are now ready to launch your insatnce! After you launch it you should be able to connect by ssh on linux or putty in windows using the key pair you exported earlier. 


### Run your app in EC2 
The final step, and the easiest, is to run your app in the EC2 instance. After connecting to the instance using [ssh](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html) or [putty](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-from-windows.html), you are ready to pull you docker image into the server. If you already pushed the Docker image to docker hub, you can simply pull using 
```bash
docker pull repo-name/real-esrgan-aws:tag   
```
Alternatively, you can clone this repo and build the image here as we explained earlier.   

After preparing your image, run the container in detach mode using the default entrypoint. This will allow us to use the terminal even after running the container

```bash
docker run -d --network host -it --gpus all repo-name/real-esrgan-aws:tag
```
Now, it is time to run the gradio demo app, for this, we will run the same container but we will override the default entrypoint which will allow us to use the container's terminal. Note that we mounted the current directory (project root) to sync changed files if any. 

```bash
docker run -it --network host --entrypoint /bin/bash -v ./:/app repo-name/real-esrgan-aws 
```

Finally, we run the gradio demo app by using the command below, You can control sharing the demo publicly by using the share parameter (default is False)

```python
python3 gradio_demo.py --share True
```




