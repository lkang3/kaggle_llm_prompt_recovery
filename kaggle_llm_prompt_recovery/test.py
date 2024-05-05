import math

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer


def preprocess_function(examples, tokenizer):
    input_text = [
        " ".join(x for x in example["text"])
        for example in examples["answers"]
    ]
    return tokenizer(input_text)


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def test():
    # https://huggingface.co/docs/transformers/en/tasks/language_modeling

    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5.cleanup_cache_files()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    tokenized_eli5 = eli5.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=4,
        remove_columns=eli5.column_names,
    )
    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
    lm_dataset = lm_dataset.train_test_split(test_size=0.2)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    training_args = TrainingArguments(
        output_dir="my_test_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    prompt = "Somatic hypermutation allows the immune system to"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    test()
