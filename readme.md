# 音樂風格分類與生成系統

這個專案包含一個基於 GTZAN 資料集訓練的音樂風格分類模型，以及一個用於音樂風格預測的系統。

## 專案結構

本專案由三個主要的 Jupyter Notebook 組成：

1.  `data_process.ipynb`: 將 GTZAN 音訊資料集處理成 26 維特徵。
2.  `classifier.ipynb`: 使用處理後的資料集訓練音樂風格分類模型。
3.  `music_gen.ipynb`: 整合訓練好的模型進行音樂風格預測，並展示了音樂生成（儘管音樂生成部分似乎是外部元件，筆記本主要專注於預測）。

## 設定與安裝

### 先決條件

* Python 3.x
* Jupyter Notebook
* 所需 Python 函式庫 (如果存在 `requirements.txt` 則可直接安裝，否則請手動安裝)：
    * `torch`
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `librosa`
    * `matplotlib`
    * `seaborn`
    * `tqdm`

您可以使用 pip 安裝這些必要的函式庫：

```bash
pip install torch pandas numpy scikit-learn librosa matplotlib seaborn tqdm
```
```bash
pip install -r requirements.txt
```

### 本專案使用 GTZAN 資料集。
請下載 GTZAN 資料集 並將 genres_wav 資料夾放置於與您的 Notebooks 相同的目錄中，或者更新 data_process.ipynb 中的 GTZAN_DATA_PATH 變數以指向正確的位置。

## 執行 Notebooks
### 1. 處理資料: 開啟並執行 data_process.ipynb。此 Notebook 將會：

* 將音訊檔案切割成 3 秒的片段。
* 從每個片段中提取 26 維特徵。
* 將處理後的資料儲存為 data_features.csv。

### 2. 訓練分類器: 開啟並執行 classifier.ipynb。此 Notebook 將會：

* 載入 data_features.csv 檔案。
* 訓練一個用於音樂風格分類的類神經網路模型。
* 評估模型的性能，並顯示混淆矩陣和各類別的準確率。

### 3. 音樂生成與預測: 開啟並執行 music_gen.ipynb。此 Notebook 將會：

* 載入訓練好的分類器模型和訓練時使用的 StandardScaler。
* 提供一個從新音訊檔案中提取特徵的函數。
* 使用訓練好的模型預測給定音訊檔案的風格，顯示預測的風格、信心度以及所有風格的機率分佈。
## Notebook 詳細說明
### data_process.ipynb
此 Notebook 負責初始資料準備。它定義了一個 GTZANFeatureExtractor 類別，該類別接收音訊檔案，將其分段，並提取各種音訊特徵（例如，MFCCs、色度特徵、頻譜對比度）。提取的特徵隨後會被編譯成一個 pandas DataFrame 並儲存為 data_features.csv CSV 檔案，該檔案將被後續的 Notebook 使用。

### classifier.ipynb
此 Notebook 專注於訓練音樂風格分類器。它使用 PyTorch 定義了一個簡單的類神經網路模型。資料從 data_features.csv 載入，被分割成訓練集和測試集，並使用 StandardScaler 和 LabelEncoder 進行預處理。然後對模型進行訓練，並使用準確率、混淆矩陣和各類別準確率評估其性能。

### music_gen.ipynb
此 Notebook 演示了如何使用訓練好的模型從新的音訊檔案中預測音樂風格。它包含了載入 StandardScaler 和訓練好的 PyTorch 模型所需的設定。提供了一個 predict_music_genre 函數，該函數接收一個音訊檔案路徑，載入它，提取特徵（使用與 data_process.ipynb 相同的邏輯），對其進行縮放，然後將其輸入到訓練好的模型中進行預測。Notebook 隨後會顯示預測的風格、信心度以及所有風格的機率分佈。

### 使用範例 (來自 music_gen.ipynb)
要預測一個新音訊檔案的風格：
```python
test_audio_path = r'genres_wav\\disco\\disco.00011.wav' # 請替換為您的音訊檔案路徑
predicted_genre, confidence, all_probs = predict_music_genre(test_audio_path, model, scaler)

if predicted_genre is not None:
    print("=== 音樂風格預測結果 ===")
    print(f"預測風格: {predicted_genre}")
    print(f"信心度: {confidence:.4f}")
    print()

    print("所有風格的機率分布:")
    for i, (genre, prob) in enumerate(zip(genres, all_probs)):
        print(f"  {genre:>10}: {prob:.4f}")
```