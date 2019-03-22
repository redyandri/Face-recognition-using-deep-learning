1. to install the opencv, just simply run:
```bash
sudo bash install_opencv.sh
```
2. to enable import cv2 from python, simply add the library of opncv to python library:
```bash
find /usr/local/lib/ -type f -name "cv2*.so"
cd ~/.virtualenvs/facecourse-py3/lib/python3.6/site-packages
ln -s /usr/local/lib/python3.6/dist-packages/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
```
3. to install tensorflow-gpu, simply run:
 ```bash
 conda install tensorflow-gpu