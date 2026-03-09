# Mini RLHF Pipeline for Language Models
This project implements a minimal Reinforcement Learning from Human Feedback (RLHF) pipeline using PyTorch.
It demonstrates how modern language models can be aligned using preference data, reward models, and policy optimization.

The project includes:

A mini language model (MiniLM) based on embeddings and LSTM

A reward model trained on human preference comparisons

A preference loss function for ranking responses

A simplified PPO-style policy update step

A custom tokenizer and vocabulary builder

Although simplified, this project illustrates the core concepts behind RLHF used in modern LLMs.

# Project Overview

Modern AI systems such as ChatGPT are trained in multiple stages:

Supervised pre-training

Reward model training

Policy optimization using reinforcement learning

This repository recreates a toy version of this pipeline for educational and research purposes.

Pipeline:

Raw Text
↓
Tokenization
↓
Vocabulary Encoding
↓
Mini Language Model
↓
Reward Model
↓
Preference Comparison
↓
Policy Optimization (PPO-style)

# Project Structure

mini-rlhf-llm/
│
├── data.py
│   Example prompts and preference pairs
│
├── tokenizer.py
│   Vocabulary builder and tokenization utilities
│
├── models.py
│   MiniLM policy model
│   RewardModel for preference scoring
│
├── rlhf.py
│   Preference loss
│   PPO-style policy update
│
├── train_reward_model.py
│   Script that evaluates preference ranking
│
├── requirements.txt
│
└── README.md


# Models
## MiniLM (Policy Model)

A lightweight language model used to generate token probabilities.

Architecture:

Embedding Layer
↓
LSTM Encoder
↓
Linear Output Layer

class MiniLM(nn.Module):

    def __init__(self, vocab, embed=128):
        super().__init__()

        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed,256,batch_first=True)
        self.fc = nn.Linear(256,vocab)

    def forward(self,x):
        x = self.embed(x)
        out,_ = self.lstm(x)
        return self.fc(out)
## Reward Model

The reward model learns to score responses based on human preferences.

Architecture:

Embedding
↓
LSTM
↓
Linear Score Output

class RewardModel(nn.Module):

    def __init__(self, vocab):
        super().__init__()

        self.embed = nn.Embedding(vocab,128)
        self.lstm = nn.LSTM(128,256,batch_first=True)
        self.score = nn.Linear(256,1)

    def forward(self,x):
        x = self.embed(x)
        out,_ = self.lstm(x)
        last = out[:,-1,:]
        return self.score(last)


## Preference Learning

The reward model is trained using pairwise preference comparisons.

Example dataset:

preferences = [

{
"prompt":"Explain phishing",
"chosen":"Phishing is a cyber attack",
"rejected":"Phishing is a type of fish"
},

{
"prompt":"What is malware",
"chosen":"Malware is malicious software",
"rejected":"Malware is a computer mouse"
}

]


## Preference Loss Function

The loss encourages the model to assign higher scores to preferred responses.

Mathematically:

𝐿=−log(𝜎(𝑟𝑐ℎ𝑜𝑠𝑒𝑛−𝑟𝑟𝑒𝑗𝑒𝑐𝑡𝑒𝑑))


Implementation:

def preference_loss(r_chosen, r_rejected):
    return -torch.log(
        torch.sigmoid(r_chosen - r_rejected)
    ).mean()

    
# PPO-Style Policy Update

A simplified reinforcement learning step is used to update the policy model.

def ppo_step(policy, reward_model, input_ids):

    logits = policy(input_ids)

    probs = torch.softmax(logits,dim=-1)

    actions = torch.multinomial(probs.view(-1,probs.size(-1)),1)

    reward = reward_model(actions)

    loss = -reward.mean()

    return loss


# Tokenization

A minimal tokenizer is implemented using:

whitespace tokenization

lowercase normalization

vocabulary mapping

Vocabulary example:

Vocabulary Size: 33


# Example Output
Prompt: Explain phishing

Chosen (score: 0.0905): Phishing is a cyber attack
Rejected (score: 0.0032): Phishing is a type of fish
Loss for this pair: 0.6505

Prompt: What is malware

Chosen (score: -0.0093): Malware is malicious software
Rejected (score: -0.0495): Malware is a computer mouse
Loss for this pair: 0.6732

Average loss of RewardModel over preferences: 0.6618


# Installation
git clone https://github.com/yourusername/mini-rlhf-llm.git

cd mini-rlhf-llm

pip install -r requirements.txt


# Requirements
torch


# Learning Goals

This project demonstrates:

Language model architectures

Tokenization pipelines

Reward modeling

Preference-based learning

Reinforcement learning for alignment

RLHF fundamentals



# Future Improvements

Possible extensions:

Transformer-based policy model

Real PPO training loop

Larger preference datasets

Beam search generation

BPE tokenizer

HuggingFace integration
