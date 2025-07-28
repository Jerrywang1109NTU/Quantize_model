cd src/Vitis-AI-Quantizer/vai_q_pytorch

if [ $? -eq 0 ]; then
  echo -e "\n#### Vai_q_pytorch code is checked out successfully."
else
  echo -e "\n#### Vai_q_pytorch code is NOT checked out successfully."
  exit 2
fi

echo -e "\n#### Installing vai_q_pytorch ..."
#pip install -r requirements.txt
cd pytorch_binding 
if [ ! -d "/usr/local/cuda" ]; then
  unset CUDA_HOME
fi
python setup.py bdist_wheel -d ./
pip install ./pytorch_nndct-*.whl
if [ $? -eq 0 ]; then
  echo -e "\n#### Vai_q_pytorch is compiled and installed successfully."
else
  echo -e "\n#### Vai_q_pytorch is NOT compiled and installed successfully."
  exit 2
fi

mamba install -y python=3.7 --force-reinstall
sudo rm -rf /scratch/*
echo -e "\n#### Cleaned up /scratch ."
