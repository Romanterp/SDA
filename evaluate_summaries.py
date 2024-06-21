from bert_score import score
import torch
from huggingface_hub import login
import transformers
from transformers import pipeline
import numpy as np
import os

# Login to Hugging Face
login(token="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")

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def load_model(model_name="FacebookAI/roberta-large"):
    config = transformers.AutoConfig.from_pretrained(model_name,
                                                     trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           truncation=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                              config=config,
                                                              torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True)
    model.to(device)
    return model, tokenizer


def setup_pipeline(model, tokenizer):
    # Set pad_token_id to eos_token_id to avoid errors in open-end generation
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    pad_token_id=tokenizer.eos_token_id)
    return pipe


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def compute_bert_score(generated_summary, reference_summary):
    P, R, F1 = score([generated_summary], [reference_summary], lang='en',
                     verbose=True, device=device)
    P_median = np.median(P.cpu().numpy())
    R_median = np.median(R.cpu().numpy())
    F1_median = np.median(F1.cpu().numpy())
    print(f"Generated Summary: {generated_summary[:100]}...")
    print(f"Reference Summary: {reference_summary[:100]}...")
    print(
        f"BERTScore - Precision: {P_median}, Recall: {R_median}, F1: {F1_median}\n")
    return P_median, R_median, F1_median


def main():
    model_name = "FacebookAI/roberta-large"
    model, tokenizer = load_model(model_name)
    pipe = setup_pipeline(model, tokenizer)

    settings = ["zero_shot", "article_top_summary", "article_all_summary",
                "article_all_summary_explanation"]
    ranks = ["Rank1", "Rank2", "Rank3"]
    articles = ["017", "018", "019", "020", "021", "022", "023", "024", "025"]

    scores = {setting: {rank: [] for rank in ranks} for setting in settings}

    for article in articles:
        for setting in settings:
            generated_summary_path = f'generated_summaries/{article}_{setting}.txt'
            if not os.path.exists(generated_summary_path):
                print(
                    f"Generated summary file {generated_summary_path} not found.")
                continue

            print(f"Reading generated summary from: {generated_summary_path}")
            generated_summary = read_text_file(generated_summary_path)
            print(
                f"Article {article}, Setting {setting}:\n{generated_summary[:100]}...\n")

            for rank in ranks:
                reference_summary_path = f'references/{article}_{rank}.txt'
                if not os.path.exists(reference_summary_path):
                    print(
                        f"Reference summary file {reference_summary_path} not found.")
                    continue

                print(
                    f"Reading reference summary from: {reference_summary_path}")
                reference_summary = read_text_file(reference_summary_path)
                P, R, F1 = compute_bert_score(generated_summary,
                                              reference_summary)
                scores[setting][rank].append(F1)

    # Calculate average F1 scores for each setting and rank
    average_scores = {
        setting: {rank: np.mean(scores[setting][rank]) for rank in ranks} for
        setting in settings}

    # Print the average scores in table format
    print(f"{'Setting':<35} {'Rank 1':<10} {'Rank 2':<10} {'Rank 3':<10}")
    for setting in settings:
        print(
            f"{setting:<35} {average_scores[setting]['Rank1']:<10.4f} {average_scores[setting]['Rank2']:<10.4f} {average_scores[setting]['Rank3']:<10.4f}")


if __name__ == "__main__":
    main()
