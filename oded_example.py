
from datasets import load_from_disk

n2c2_dataset_path = 'n2c2_dataset'
n2c2_dataset = load_from_disk(n2c2_dataset_path)

# ----------------------------- NER task ------------------------------------------------------------

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pubmedbert = AutoModel.from_pretrained(model_name)

# Load datalaoders
medical_ner_dataset = MedicalNERDataset(n2c2_dataset, tokenizer)
ner_dataloaders = medical_ner_dataset.get_dataloaders()
ner_train_dataloader = ner_dataloaders["train"]
ner_val_dataloader = ner_dataloaders["validation"]

# Load model
crf_model = NERHead(num_labels=len(medical_ner_dataset.id_2_label))

for batch in ner_train_dataloader:
    loss = crf_model(inputs=output, batch)
    break


# ----------------------------- RC task ------------------------------------------------------------

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pubmedbert = AutoModel.from_pretrained(model_name)

# Load datalaoders
medical_rc_dataset = MedicalRCDataset(n2c2_dataset, tokenizer, pubmedbert)
rc_dataloaders = medical_rc_dataset.get_dataloaders()
rc_train_dataloader = rc_dataloaders["train"]
rc_val_dataloader = rc_dataloaders["validation"]

# Load model
RC_model = RelationClassificationHead()

for batch in rc_train_dataloader:
    preds = RC_model(inputs=output, batch)
    break