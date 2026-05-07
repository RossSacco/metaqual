# Fine-Tuning T5 Query-Independent (stile QualT5)

Trainer principale:
- `models/qualt5_ft.py`

Supporta due sorgenti triple:
- `--triples_source file` (default, retrocompatibile)
- `--triples_source irds` (MSMARCO via ir_datasets)

## Training locale

```bash
python -m models.qualt5_ft \
  --triples_source file \
  --model_name_or_path t5-base \
  --triples_path /data/msmarco/triples.train.small.tsv \
  --triples_format text \
  --output_dir /data/models/qualt5-msmarco-quality \
  --max_steps 10000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --bf16
```

## Training con triples a ID (collection.tsv)

```bash
python -m models.qualt5_ft \
  --triples_source file \
  --model_name_or_path t5-base \
  --triples_path /data/msmarco/qidpidtriples.train.full.2.tsv.gz \
  --triples_format id \
  --collection_path /data/msmarco/collection.tsv \
  --output_dir /data/models/qualt5-msmarco-quality \
  --max_steps 10000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --bf16
```

## Training con triples a ID via DatasetLoader MSMARCO

```bash
python -m models.qualt5_ft \
  --triples_source file \
  --model_name_or_path t5-base \
  --triples_path /data/msmarco/qidpidtriples.train.full.2.tsv.gz \
  --triples_format id \
  --dataset_name msmarco_passage \
  --output_dir /data/models/qualt5-msmarco-quality \
  --max_steps 10000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --bf16
```

## Training via ir_datasets (senza triples locali)

```bash
python -m models.qualt5_ft \
  --triples_source irds \
  --irds_dataset_id msmarco-passage/train/triples-small \
  --model_name_or_path t5-base \
  --output_dir outputs/qt5-supervised-t5-base \
  --max_steps 10000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --max_length 256 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --logging_steps 50 \
  --fp16
```

## Training + upload su Hugging Face

```bash
python -m models.qualt5_ft \
  --triples_source file \
  --model_name_or_path t5-base \
  --triples_path /data/msmarco/triples.train.small.tsv \
  --triples_format text \
  --output_dir /data/models/qualt5-msmarco-quality \
  --hub_model_id RossSacco/qualt5-msmarco-quality \
  --push_to_hub \
  --hub_private_repo \
  --bf16
```

## Resume da checkpoint

```bash
python -m models.qualt5_ft \
  --triples_source file \
  --model_name_or_path t5-base \
  --triples_path /data/msmarco/triples.train.small.tsv \
  --triples_format text \
  --output_dir /data/models/qualt5-msmarco-quality \
  --resume_from_checkpoint /data/models/qualt5-msmarco-quality/checkpoint-5000 \
  --bf16
```

## Help

```bash
python -m models.qualt5_ft --help
```
