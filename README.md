# Classificate_Disaster_Msg
Classifies disaster messages by category.

# Prepare the dataset
1. Place the file `data.tsv` in the same folder as the `train.py` file.
   * The `data.tsv` file should have a 'MSG_CN' column and a 'DSSTR_SE_NM' column. The 'MSG_CN' is the disaster messages text, and the 'DSSTR_SE_NM' enters the categorization value of the Disaster messages text (MSG_CN).

2. Install the necessary packages.
```shell
pip install torch pandas numpy scikit-learn tqdm
```

   * `train.py` will train using a GPU if it is available. If your environment is not set up for GPU usage, please install PyTorch with cuda from the [PyTorch Official Website](https://pytorch.org/get-started/locally) in the `Compute Platform` section. Training on a CPU will be very slow.

3. Execute the `train.py` file.

# Testing a trained model
You can use the learned model to check for malicious comments. When you run the test.py file, type the desired disaster message text after and run it, and you'll get the result. Example:

```bash
python test.py "현재 호우주의보가 발효중이오니 야외활동 및 외출 자제, 하천저지대·산사태 우려지역 등 위험지역 접근을 금지하시고 안전에 유의하시기 바랍니다."
# or
python3 test.py "현재 호우주의보가 발효중이오니 야외활동 및 외출 자제, 하천저지대·산사태 우려지역 등 위험지역 접근을 금지하시고 안전에 유의하시기 바랍니다."
```

# Used Model
Training is conducted using the [KoELECTRA model (KoELECTRA-Base-v3)](https://github.com/monologg/KoELECTRA).

# loss/accuracy/f1-score value when learning AI
The loss/accuracy/f1-score values from AI learning are as follows:

<img src="./disaster_msg_classification_ai_graph.png" width="500">

* **Epoch 1/3**
  - Train Loss: 0.3961016160589645, Train Accuracy: 0.9176465107612037, Train F1 Score: 0.9170265443520098
  - Val Loss: 0.09513981542512376, Val Accuracy: 0.9762277440813931, Val F1 Score: 0.9759060699969906

* **Epoch 2/3**
  - Train Loss: 0.07227880962242708, Train Accuracy: 0.9807323208795304, Train F1 Score: 0.9805661145222697
  - Val Loss: 0.06824474746271376, Val Accuracy: 0.9834996412965499, Val F1 Score: 0.9833438367317662

* **Epoch 3/3**
  - Train Loss: 0.04191096748945087, Train Accuracy: 0.9894158203670922, Train F1 Score: 0.9893351303139123
  - Val Loss: 0.05774910528889658, Val Accuracy: 0.9863692688971499, Val F1 Score: 0.9862527352124375

* **Test Loss**: 0.05175440165003503, Test Accuracy: 0.9873475510337181, Test F1 Score: 0.9872944783691321