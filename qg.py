import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os


class CustomDataset(Dataset):

  def __init__(self, data, tokenizer, source_len, summ_len):
    self.tokenizer = tokenizer
    self.data = data
    self.source_len = source_len
    self.summ_len = summ_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    example = self.data[index]
    snippet = example['snippet'].replace('\n', ' ')
    inp = f"answer: {example['explain']} </s> context: {snippet}"
    answer = example['answer']
    source = self.tokenizer(inp, max_length=self.source_len, padding='max_length',
                            truncation=True, return_tensors='pt')

    target = self.tokenizer(answer, max_length=self.summ_len, padding='max_length',
                            truncation=True, return_tensors='pt')

    example['source_ids'] = source['input_ids'].squeeze()
    example['source_mask'] = source['attention_mask'].squeeze()
    example['target_ids'] = target['input_ids'].squeeze()
    example['target_mask'] = target['attention_mask'].squeeze()
    return example


def train(epoch, tokenizer, model, device, loader, optimizer):
  model.train()
  for _, data in enumerate(tqdm(loader)):
    ids = data['source_ids'].squeeze(0).to(device, dtype=torch.long)
    mask = data['source_mask'].squeeze(0).to(device, dtype=torch.long)
    lm_labels = data['target_ids'].squeeze(0).to(device)
    decoder_attention_mask = data['target_mask'].squeeze(0).to(device)
    lm_labels[lm_labels == tokenizer.pad_token_id] = -100
    outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
    loss = outputs[0]

    if _ % 500 == 0:
      print(f'Epoch: {epoch}, Loss:  {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def validate(epoch, tokenizer, model, device, loader):
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
    for _, data in enumerate(tqdm(loader)):
      y = data['target_ids'].to(device, dtype=torch.long)
      ids = data['source_ids'].to(device, dtype=torch.long)
      mask = data['source_mask'].to(device, dtype=torch.long)
      generated_ids = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
      )
      preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
      target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
      if _ % 100 == 0:
        print(f'Completed {_}')
      predictions.extend(preds)
      actuals.extend(target)
  return predictions, actuals


def collate_data(batch):
  target_ids = torch.stack([i['target_ids'] for i in batch])
  target_mask = torch.stack([i['target_mask'] for i in batch])
  source_ids = torch.stack([i['source_ids'] for i in batch])
  source_mask = torch.stack([i['source_mask'] for i in batch])

  return {
    'target_ids': target_ids,
    'target_mask': target_mask,
    'source_ids': source_ids,
    'source_mask': source_mask
  }


def main():
  # Defining some key variables that will be used later on in the training
  TRAIN_BATCH_SIZE = 8  # input batch size for training (default: 64)
  VALID_BATCH_SIZE = 4  # input batch size for testing (default: 1000)
  TRAIN_EPOCHS = 10  # number of epochs to train (default: 10)
  VAL_EPOCHS = 1
  LEARNING_RATE = 2e-4  # learning rate (default: 0.01)
  SEED = 323  # random seed (default: 42)
  MAX_LEN = 300
  SUMMARY_LEN = 100
  MODEL = 't5-base'
  # Set random seeds and deterministic pytorch for reproducibility
  torch.manual_seed(SEED)  # pytorch random seed
  np.random.seed(SEED)  # numpy random seed
  torch.backends.cudnn.deterministic = True

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("DEVICE", device)
  # tokenzier for encoding the text
  torch.cuda.empty_cache()
  tokenizer = T5Tokenizer.from_pretrained(MODEL, model_max_length=MAX_LEN, truncation=True)
  train_path = os.path.join(os.getcwd(), "data/sharc_raw/json/sharc_train_question_fixed.json")
  val_path = os.path.join(os.getcwd(), "data/sharc_raw/json/sharc_dev_question_fixed.json")
  explain_path = os.path.join(os.getcwd(), "data/explanations.json")
  train_dataset = json.load(open(train_path))
  val_dataset = json.load(open(val_path))
  parsed_train_dataset = []
  parsed_val_dataset = []
  explain_dataset = json.load(open(explain_path))
  for i in train_dataset:
    answer = i['answer'] if i['answer'] != 'Yes?' else 'Yes'
    if answer != 'Yes' and answer != 'No' and answer != 'Irrelevant':
      i['explain'] = ' '.join(explain_dataset[i['tree_id']]['questions'][answer])
      parsed_train_dataset.append(i)

  for i in val_dataset:
    answer = i['answer'] if i['answer'] != 'Yes?' else 'Yes'
    if answer != 'Yes' and answer != 'No' and answer != 'Irrelevant':
      i['explain'] = ' '.join(explain_dataset[i['tree_id']]['questions'][answer])
      parsed_val_dataset.append(i)
  # Creating the Training and Validation dataset for further creation of Dataloader
  training_set = CustomDataset(parsed_train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
  val_set = CustomDataset(parsed_val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

  # Defining the parameters for creation of dataloaders
  train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
  }

  val_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
  }

  # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
  training_loader = DataLoader(training_set, collate_fn=collate_data, **train_params)
  val_loader = DataLoader(val_set, collate_fn=collate_data, **val_params)

  # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
  # Further this model is sent to device (GPU/TPU) for using the hardware.
  model = T5ForConditionalGeneration.from_pretrained(MODEL)
  model = model.to(device)

  # Defining the optimizer that will be used to tune the weights of the network in the training session.
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


  # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
  # Saving the dataframe as predictions.csv
  print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
  resume_dir = os.path.join(os.getcwd(), 'out/t5_squad_5_epoch_batch_8_seed_323.pt')
  qg_dir = os.path.join(os.getcwd(), 'out/t5_squad_10_epoch_batch_8_seed_323.csv')

  for epoch in range(TRAIN_EPOCHS):
      train(epoch, tokenizer, model, device, training_loader, optimizer)
  for epoch in range(VAL_EPOCHS):
    predictions, actual = validate(epoch, tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actual})
    with open(qg_dir, 'w') as f:
      final_df.to_csv(f)
    print('Output Files generated for review')


if __name__ == '__main__':
  main()
