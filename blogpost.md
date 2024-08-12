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



