# training the model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
def train_model(model_name, train_dataset, output_dir, num_train_epochs=3, per_device_train_batch_size=4):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=500,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()