# Captcha_Datasets
Data sets for captchas.

Current scripts:
```
    src/
        ai_faces.py
        ai_images.py
        web_images.py
```

## 📌 Scripts in the future
- Generating tones and distorted / not distorted music
- Generating sound splitters
- Emoji dataset using images of Emoji versions over the time

## Creating your own data sets
Please create your own data sets before using them with [tn3w/flask_Captchaify](https://github.com/tn3w/flask_Captchaify). You could use the default data sets, however these are not very large, a size of at least 1 GB is recommended.

### 🚀 Installing
#### Clone the Repository
```bash
git clone https://github.com/tn3w/Captcha_Datasets
```

or download the zip [here](https://github.com/tn3w/Captcha_Datasets/archive/refs/heads/master.zip).

#### Create Venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### (Optional) Installing Cuda
CUDA makes the code run faster because your graphics card can also be used.

##### Ubuntu/Debian-based Systems
Add NVIDIA Package Repository:
```bash
sudo apt update
sudo apt install -y build-essential
sudo apt install -y linux-headers-$(uname -r)
```

Download .deb from [here](https://developer.nvidia.com/cuda-downloads).

Install it using:
```bash
sudo dpkg -i cuda-repo-<distro>_<version>_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$(lsb_release -si | tr '[:upper:]' '[:lower:]')/$(lsb_release -sc)/7fa2af80.pub
sudo apt update
sudo apt install -y cuda
```

##### Fedora
Add NVIDIA Package Repository:
```bash
sudo dnf install -y kernel-devel kernel-headers
```

Download .rpm from [here](https://developer.nvidia.com/cuda-downloads).

Install it using:
```bash
sudo rpm -i cuda-repo-<distro>_<version>_x86_64.rpm
sudo dnf clean expire-cache
sudo dnf install -y cuda
```

##### Arch
```bash
sudo pacman -Syu cuda
```

##### CentOS/RHEL
Add NVIDIA Package Repository:
```bash
sudo yum install -y epel-release
```

Download .rpm from [here](https://developer.nvidia.com/cuda-downloads).

Install it using:
```bash
sudo rpm -i cuda-repo-<distro>_<version>_x86_64.rpm
sudo yum clean expire-cache
sudo yum install -y cuda
```

##### macOS
Download and install by downloading it [here](https://developer.nvidia.com/cuda-downloads)

##### Windows
Download and install by downloading it [here](https://developer.nvidia.com/cuda-downloads)

#### Docker
Install NVIDIA Docker using the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Run a CUDA Container:
```bash
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### (Optional) Installing OpenGL
If you encounter the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, you can resolve it by installing the OpenGL package. Below are the commands specific to your operating system:

##### Ubuntu/Debian-based Systems
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

##### Fedora
```bash
sudo dnf install mesa-libGL
```

##### Arch Linux
```bash
sudo pacman -S mesa
```

##### CentOS/RHEL
```bash
sudo yum install mesa-libGL
```

##### macOS
OpenGL is included with macOS, but you can ensure your system is up to date. If you are using Homebrew, you can reinstall OpenCV:
```bash
brew install opencv
```

##### Windows
For Windows, ensure that you have the appropriate OpenGL drivers installed. You can usually download these from your graphics card manufacturer's website (NVIDIA, AMD, Intel).

##### Docker
If you are using a Docker container based on Ubuntu, you can add the following line to your Dockerfile:
```Dockerfile
RUN apt-get update && apt-get install -y libgl1-mesa-glx
```