# 基于T5的中文问答生成模型

基于T5预训练模型实现的中文问答生成任务，使用百度开源的DuReaderQG数据集进行训练。该模型能够根据输入的问题和上下文生成相应的答案。

## 项目特点

- 使用T5-base-chinese作为基础模型
- 支持中文问答生成
- 提供完整的训练和推理代码
- 包含详细的性能评估（BLEU指标）
- 支持自定义数据集训练

## 环境要求

- Python 3.6+
- PyTorch
- Transformers
- Rich
- Datasets
- numpy
- matplotlib

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

项目使用百度开源的DuReaderQG数据集，数据格式如下：

```json
{
    "context": "违规分为:一般违规扣分、严重违规扣分、出售假冒商品违规扣分,淘宝网每年12月31日24:00点会对符合条件的扣分做清零处理...",
    "answer": "12月31日24:00",
    "question": "淘宝扣分什么时候清零",
    "id": 203
}
```

数据字段说明：
- context: 文章内容
- question: 问题
- answer: 答案
- id: 样本编号

## 模型训练

### 训练参数说明

修改 `train.sh` 中的参数进行训练：

```bash
python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/DuReaderQG" \
    --train_path "data/DuReaderQG/train.json" \
    --dev_path "data/DuReaderQG/dev.json" \
    --img_log_dir "logs/DuReaderQG" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 32 \
    --max_source_seq_len 256 \
    --max_target_seq_len 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --valid_steps 500 \
    --device "cuda:0"
```

主要参数说明：
- pretrained_model: 预训练模型路径
- save_dir: 模型保存路径
- train_path: 训练集路径
- dev_path: 验证集路径
- batch_size: 批次大小
- learning_rate: 学习率
- num_train_epochs: 训练轮数
- device: 训练设备

### 训练过程

训练过程中会显示如下信息：
```
global step 10, epoch: 1, loss: 9.39613, speed: 1.60 step/s
global step 20, epoch: 1, loss: 9.39434, speed: 1.71 step/s
...
```

同时会在logs目录下生成训练曲线图，包含：
- 训练损失曲线
- BLEU-1/2/3/4评估指标曲线

## 模型推理

使用训练好的模型进行推理：

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained('checkpoints/DuReaderQG/model_best/')
model = T5ForConditionalGeneration.from_pretrained('checkpoints/DuReaderQG/model_best/')

# 示例推理
question = '治疗宫颈糜烂的最佳时间'
context = '专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面...'
input_seq = f'问题：{question}{tokenizer.sep_token}原文：{context}'

# 生成答案
inputs = tokenizer(input_seq, return_tensors='pt', max_length=256, truncation=True)
outputs = model.generate(input_ids=inputs["input_ids"])
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 评估指标

模型使用BLEU-1/2/3/4作为评估指标，评估代码位于`bleu_metrics.py`。

## 项目结构

```
.
├── train.py          # 训练脚本
├── inference.py      # 推理脚本
├── train.sh          # 训练启动脚本
├── requirements.txt  # 项目依赖
├── utils.py          # 工具函数
├── bleu_metrics.py   # BLEU评估指标
├── iTrainingLogger.py# 训练日志记录器
├── data/             # 数据目录
│   └── DuReaderQG/  # 问答数据集
├── logs/            # 日志目录
└── checkpoints/     # 模型保存目录
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交问题和PR！

## 致谢

- 感谢百度开源的DuReaderQG数据集
- 感谢UER团队开源的中文T5预训练模型

