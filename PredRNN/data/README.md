# 数据放置说明

仓库不包含 CIKM2017 原始数据。请解压你已有的三个 `.tar.gz` 文件，将内部目录放在以下位置：

```text
data/raw/data_new/
├── CIKM2017_train/train.txt
├── CIKM2017_testA/testA.txt
├── CIKM2017_testA/answer_sample.csv
├── CIKM2017_testB/testB.txt
└── CIKM2017_testB/answer_sample.csv
```

例如在项目根目录运行：

```bash
mkdir -p data/raw
tar -xzf /path/to/CIKM2017_train.tar.gz -C data/raw
tar -xzf /path/to/CIKM2017_testA.tar.gz -C data/raw
tar -xzf /path/to/CIKM2017_testB.tar.gz -C data/raw
```

训练代码逐条读取大型文本记录，并把偏移索引写到 `outputs/index/`。这些原始文件和索引都由 `.gitignore` 排除，不会意外推送到 GitHub。
