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

---

### Unimodal Encoders Location

The **unimodal encoders** (for text, audio, and visual) are defined in `mm.py` file
with class `UnimodalEncoder`, which is a general-purpose wrapper around an 
attention-augmented LSTM/GRU (`AttentiveRNN`).
`UnimodalEncoder` is used by:

* `AVTEncoder`, is the multimodal encoder that combines three `UnimodalEncoder` instances 
and then fuses them via `AttRnnFuser`. 
* `AVTClassifier`, is the final classification model that adds a linear layer on top 
of the `AVTEncoder` for classification.

Each modality is handled via a separate instance:

```python
self.text = UnimodalEncoder(...)
self.audio = UnimodalEncoder(...)
self.visual = UnimodalEncoder(...)
```

---

### Final LSTM (Fusion Layer)

The final LSTM after unimodal encoding is in file `mm.py` with classname `AttRnnFuser`.
This fuses all three modalities:

```python
self.rnn = AttentiveRNN(...)
```

`AttentiveRNN` wraps around a bidirectional LSTM and adds attention:

* Defined in `rnn.py`
* Class: `AttentiveRNN`
* Internally uses `RNN` class with `rnn_type="lstm"`

This fused representation is the final output passed to the classifier.

---

So, every LSTM in the model is implemented via `AttentiveRNN`. Each modality in `UnimodalEncoder` 
and the final Fuser LSTM: `AttRnnFuser` all use `AttentiveRNN`, a regular LSTM and adds optional attention 
on top of the LSTM output.