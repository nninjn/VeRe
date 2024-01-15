# VeRe: Verification Guided Synthesis for Repairing Deep Neural Networks

This repository contains full experimental results for the paper "VeRe: Verification Guided Synthesis for Repairing Deep Neural Networks" in ICSE 2024.

## Prerequisites
- We provide a docker image for Linux that has already been initialized with all dependencies, which contains all models and datasets. Our image can be obtained from: [Figshare](https://figshare.com/articles/software/Artifact_VeRe_Verification_Guided_Synthesis_for_Repairing_Deep_Neural_Networks/24920130). 
- VeRe requires a GPU for calculating repair significance and fixing problematic neurons, so the Nvidia-docker2 is required for Docker. The tutorial can be found [here](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/). Ensure that your Docker version is not higher than 19.03.
- Ensure that your CUDA version is greater than or equal to 12.0 to successfully run the image.

## Usage
- The Docker image is also available on Docker Hub. To obtain our image, utilize the following command: sudo docker pull mjnn/vere:1.0.
- If you download an image from Figshare, you can use the following command to load the image `sudo docker load -i vere.tar`.
- After obtaining the image, the first step is to import the Docker image from the tarball and create a corresponding container to run VeRe `sudo docker run --runtime=nvidia -it --name Artifact-vere mjnn/vere:1.0`.
- Please read the `README_Artifact.pdf` for details.

## Files
**`backdoor`: Experiments for Backdoor Removal**

**`safety`: Experiments for Correcting Safety Property**

**`results`: Full Experimental Results**


