## Install pytorch and realted packages
## We suggest pytorch version above 1.7.0, best with 1.9.0
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge


pip install PyYAML
pip install easydict
pip install cython
pip install opencv-python==4.1.0.25
pip install opencv-contrib-python==4.1.0.25
pip install pycocotools

conda install -y tqdm
apt-get install libturbojpeg
pip install jpeg4py
pip install tb-nightly
pip install pandas
pip install timm


## install openvino and onnx
## https://github.com/intel/onnxruntime/releases/tag/v3.1
pip install onnx
pip install onnxruntime_openvino-1.9.0-cp37-cp37m-linux_x86_64.whl #(change to corresponding version)


