==== Prerequisites ====

The program was tested under and is compatible with:
Python 3.6.6
PyTorch 1.0.1.post2
CentOS 7 x86\_64

It may work with other versions of Python 3, PyTorch and operating systems
as well, but PyTorch is known to break compatibilities 
between versions.
It definitely does not work with Python 2.
As long as it runs without failure, the program should 
work correctly.

The program depends on pip packages:
torch (not directly installable from pip repositories)
numpy (1.16.2)
Pillow (5.4.1)
torchvision (0.2.2.post3)
matplotlib (3.0.3)

Other versions from pip or PIL instead of Pillow should work,
except that torchvision from PyTorch may have bad compatibilities between versions.
As long as it runs without failure, the program should 
work correctly.

To install PyTorch 1.0.1.post2 under Linux, run:
pip3.6 install --user https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3.6 install --user torchvision

To install latest stable PyTorch under MacOS, run:
pip3 install torch torchvision
Unfortunately, PyTorch are not well-supported on MacOS. 
GPU is not supported and the installation may even break.
Try "import torch" on the REPL and ensure that PyTorch works correctly before 
proceeding.

To install PyTorch under Windows:
pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.1.post2-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu90/torchvision-0.2.2.post3-cp37-cp37m-win_amd64.whl

==== Images ====

The program only takes square (same width and height) JPEG
images whose file name ends with ".jpg".
The images do not have to be of the same size as long as they are all sqaure.
They will be automatically scaled.
The image files must be directly under the current working directory,
which is usually the same place as the neural-style.py file.
It has been designed this way so that it can generate 
multiple image files according to the filenames of the
style and content images.

==== How to Run ====

To run the program with the style image "starrynight.jpg" and 
content image "dancing.jpg" for example, run:
python36 neural-style.py --one dancing starrynight

The images after each 50 iterations will then be written
to the working directory with names like dancing-starrynight-*-????.png,
where the ???? are number of iterations.
There will also be a dancing-starrynight-*.txt to record the losses.

The image size is hardcoded to 512x512 in the python code 
so it may run slowly (dozens of minutes) without GPU.
Alternatively you can add the image size after the two image names,
like:
python36 neural-style.py --one dancing starrynight 128

==== Credits ====
The program was adapted from 
https://pytorch.org/tutorials/_downloads/7d103bc16c40d35006cd24e65cf978d0/neural_style_tutorial.py
