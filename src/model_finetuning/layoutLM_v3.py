from train_dataset_prep import TRAIN_DATA
from test_dataset_prep import TEST_DATA
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from dataset_util import preprocess

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

# Preprocess both datasets for the training and testing
ENCODED_TRAIN_DATASET = {}
for sample in TRAIN_DATA:
    encodings = preprocess(sample, label2id, processor)

