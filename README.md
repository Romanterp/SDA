# SDA
Scholarly Document Analysis 

Introduction

This project utilizes Hugging Face's Transformers library to generate summaries for articles using different settings and compares these generated summaries to reference summaries using BERTScore. The goal is to determine which setting produces summaries that are most similar to highly-ranked summaries.
Features

    Summary Generation: Generate summaries for articles using different settings.
    Summary Comparison: Compare generated summaries to reference summaries using BERTScore.
    Rank Evaluation: Evaluate and rank the performance of different summary generation settings.

Usage

    Login to Hugging Face: Ensure you have a Hugging Face token. Replace your_huggingface_token with your actual token in the script.

python

login(token="your_huggingface_token")

    Generate Summaries: Run the script to generate summaries for all articles in the summaries directory. The generated summaries will be saved in the references directory.

python

python generate_summaries.py

    Evaluate Summaries: Run the evaluation script to compare the generated summaries with the reference summaries using BERTScore.

python

python evaluate_summaries.py
