# mmlatch

<pre lang="bash">
```bash
conda env create -f environment.yml
conda activate mmlatch
PYTHONPATH=$(pwd)/cmusdk:$(pwd)

git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK
pip install .
pip install -e .

pip install numpy==1.24.4
pip install validators==0.18
cd ..
python run_original_mosei.py --config config.yaml
```
</pre>
