from train_dataset_prep import TRAIN_DATA
from test_dataset_prep import TEST_DATA
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from dataset_util import preprocess
from datasets import Dataset
from path_config import PROCESSED_DATA_PATH
from loguru import logger
import os
import torch



# Collect unique labels from the datasets, so that we can assign them numbers (ids)
unique_labels = set()
for sample in TRAIN_DATA:
    unique_labels.update(sample['labels'])
for sample in TEST_DATA:
    unique_labels.update(sample['labels'])
unique_labels = sorted(list(unique_labels))

# assign IDs to the labels & create mappings for model input
label2id = {k:v for v, k in enumerate(unique_labels)}
id2label = {k:v for v, k in label2id.items()}

# Load processor (It combines image processor and tokenizer).
processor = LayoutLMv3Processor.from_pretrained( "microsoft/layoutlmv3-base", apply_ocr=False)
# Load base model
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

ENCODED_TRAIN_DATASET = None
ENCODED_TEST_DATASET = None
encoded_train_dataset_path = PROCESSED_DATA_PATH / 'ENCODED_TRAIN_DATASET.pt'
encoded_test_dataset_path = PROCESSED_DATA_PATH / 'ENCODED_TEST_DATASET.pt'
if os.path.exists(encoded_train_dataset_path) and  os.path.exists(encoded_test_dataset_path):
    ENCODED_TRAIN_DATASET = torch.load(encoded_train_dataset_path)
    logger.info("ENCODED_TRAIN_DATASET.pt loaded!")
    ENCODED_TEST_DATASET = torch.load(encoded_test_dataset_path)
    logger.info("ENCODED_TRAIN_DATASET.pt loaded!")
else:
    # Preprocess training dataset
    ENCODED_TRAIN_DATASET = {'input_ids': [], 'attention_mask': [], 'bbox': [], 'labels': [], 'pixel_values': []}
    for sample in TRAIN_DATA:
        encodings = preprocess(sample, label2id, processor)
        ENCODED_TRAIN_DATASET['input_ids'].append(encodings['input_ids'])
        ENCODED_TRAIN_DATASET['attention_mask'].append(encodings['attention_mask'])
        ENCODED_TRAIN_DATASET['bbox'].append(encodings['bbox'])
        ENCODED_TRAIN_DATASET['labels'].append(encodings['labels'])
        ENCODED_TRAIN_DATASET['pixel_values'].append(encodings['pixel_values'])
    torch.save(ENCODED_TRAIN_DATASET, encoded_train_dataset_path)
    logger.success(f"ENCODED_TRAIN_DATASET.pt successfully created!")

    # Preprocess testing dataset
    ENCODED_TEST_DATASET = {'input_ids': [], 'attention_mask': [], 'bbox': [], 'labels': [], 'pixel_values': []}
    for sample in TEST_DATA:
        encodings = preprocess(sample, label2id, processor)
        ENCODED_TEST_DATASET['input_ids'].append(encodings['input_ids'])
        ENCODED_TEST_DATASET['attention_mask'].append(encodings['attention_mask'])
        ENCODED_TEST_DATASET['bbox'].append(encodings['bbox'])
        ENCODED_TEST_DATASET['labels'].append(encodings['labels'])
        ENCODED_TEST_DATASET['pixel_values'].append(encodings['pixel_values'])
    torch.save(ENCODED_TEST_DATASET, encoded_test_dataset_path)
    logger.success(f"ENCODED_TEST_DATASET.pt successfully created!")

ENCODED_TRAIN_DATASET = Dataset.from_dict(ENCODED_TRAIN_DATASET)
ENCODED_TEST_DATASET = Dataset.from_dict(ENCODED_TEST_DATASET)

# Training & Testing Dataset
'''
Dataset({
    features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
    num_rows: 626
})
Dataset({
    features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
    num_rows: 345
})
'''

