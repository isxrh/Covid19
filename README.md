# 面向COVID-19检测模型的语音对抗样本攻击
## 使用指北

### 0. 数据准备
- 下载数据集（https://github.com/iiscleap/Coswara-Data）
- Coswara-Data项目中extract_data.py方法解压数据集提取出Extracted_data文件
- 运行preparing_coswara_data.py处理原始数据
    ```shell
    python preparing_coswara_data.py
    ```
- 运行data_processing.py进行数据处理(去静音段、数据增强、提取语谱图特征)
    ```shell
    python preparing_coswara_data.py
    ```
  
### 1. 检测模型

- ResNet18:
  ```shell
  python spectrogram.py ResNet18
  ```

- ResNet50:
  ```shell
  python spectrogram.py ResNet50
  ```

- Cider:
  ```shell
  python spectrogram.py Cider
  ```

### 2. 生成对抗样本

- FGSM
  - ResNet18
    ```shell
    python ae.py ResNet18 fgsm
    ```
  - ResNet50
    ```shell
    python ae.py ResNet50 fgsm
    ```
  - ResNet18
    ```shell
    python ae.py Cider fgsm
    ```

- PGD
  - ResNet18
    ```shell
    python ae.py ResNet18 pgd
    ```
  - ResNet50
    ```shell
    python ae.py ResNet50 pgd
    ```
  - ResNet18
    ```shell
    python ae.py Cider pgd
    ```

### 3. 防御

- 输入变换（时间偏移）

```shell
python defense_timeshift.py ResNet18 fgsm
python defense_timeshift.py ResNet50 fgsm
python defense_timeshift.py Cider fgsm
python defense_timeshift.py ResNet18 pgd
python defense_timeshift.py ResNet50 pgd
python defense_timeshift.py Cider pgd
```

- 对抗训练
```shell
python defense_attacktrain.py ResNet18
python defense_attacktrain.py ResNet50
python defense_attacktrain.py Cider
```

### 4. 论文绘图

- `plot_figure.ipynb`