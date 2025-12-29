# MIR-2025-Autumn

## 环境搭建
```shell
# environment setup
conda create -n mir2025 python=3.11
cd MIR-2025-Autumn
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# install jdk, Lucene needs
sudo apt install openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```
## 实验
### DPR

相关权重在[Modelscope](https://modelscope.cn/models/modeledom/2025_ir_dpr/)开源。

```shell
# train encoder
python DPR/train_dense_encoder.py \
    datasets=hw \
    train_datasets=[path to webq_train] \
    dev_datasets=[path to webq_dev] \
    train=biencoder_tiny \
    output_dir=weights \
    encoder=hf_bert

# encode corpus
python DPR/generate_dense_embeddings.py \
	model_file=\'$(pwd)/output/webq/dpr_biencoder_best.pt\' \
	ctx_src=[path to wiki_webq_corpus] \
	batch_size=128 \
	out_file='wiki_webq_corpus.pkl'
	
# retrieve
python DPR/dense_retriever.py \
	model_file=\'$(pwd)/output/webq/dpr_biencoder_best.pt\' \
	qa_dataset=[path to webq_test] \
	ctx_datatsets=[path to wiki_webq_corpus] \
	encoded_ctx_files=[\"$(pwd)/output/webq/wiki_webq_corpus.pkl\"] \
	out_file='output/result_dpr.json' \

# evaluate
export PYTHONPATH=$(pwd)/DPR
python cal_hit_multi.py -q datas/webq-test.csv -r output/result_dpr.json -n dpr -t 10
```

### BM25 and HyDE
流程中需要使用 DeepSeek API，代码中已将 API Key 进行硬编码(`bm25/utils.LMAPI`)，可根据需要自行更改。

```shell
cd MIR-2025-Autumn
# build bm25 index
python bm25/bm25.py --build_index --corpus_path [path to wiki_webq_corpus] --index_dir bm25/bm25_index/

# retrieve with bm25 and hyde
python bm25/bm25.py --queries_path datas/webq-test.txt --index_dir bm25/bm25_index/ --top_k 100 --output output/result_bm25.json
python bm25/bm25.py --queries_path datas/webq-test.txt --index_dir bm25/bm25_index/ --top_k 100 --output output/result_bm25+hyde.json --hyde --hyde_concat

# evaluate
export PYTHONPATH=$(pwd)/DPR
python cal_hit_multi.py -q datas/webq-test.csv -r output/result_bm25.json -n bm25 -t 10
python cal_hit_multi.py -q datas/webq-test.csv -r output/result_bm25+hyde.json -n bm25+hyde -t 10
```

### Statistic
```shell
cd MIR-2025-Autumn
python statistic.py
```

