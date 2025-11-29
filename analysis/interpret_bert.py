import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

def compute_saliency(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # we need the embeddings to compute gradients relative to them
    # access the embedding layer directly
    embedding_layer = model.bert.embeddings.word_embeddings
    
    # create input embeddings manually so we can attach a hook/require grad
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    
    input_embeddings = embedding_layer(input_ids).clone().detach()
    input_embeddings.requires_grad = True
    
    outputs = model(inputs_embeds=input_embeddings, 
                    token_type_ids=token_type_ids, 
                    attention_mask=attention_mask)
    
    score = outputs.logits[0][0]

    score.backward()
    
    # Saliency = Dot product of embeddings and their gradients (Input * Gradient)
    # "attribution" of the score for each token
    gradients = input_embeddings.grad[0]
    attributions = torch.sum(input_embeddings[0] * gradients, dim=1)
    
    # Normalize for visualization
    attributions = attributions.detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return tokens, attributions, score.item()

def print_analysis(tokens, attributions, score):
    print(f"\nPredicted Log Score: {score:.4f} (Exp Score: {np.expm1(score):.2f})")
    
    # filter out special tokens for cleaner output
    scored_tokens = []
    for t, attr in zip(tokens, attributions):
        if t not in ['[CLS]', '[SEP]', '[PAD]']:
            scored_tokens.append((t, attr))
            
    scored_tokens.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 words pushing score UP (+):")
    for t, attr in scored_tokens[:5]:
        print(f"  {t:<15} ({attr:+.4f})")
        
    print("Top 5 words pushing score DOWN (-):")
    for t, attr in scored_tokens[-5:][::-1]:
        print(f"  {t:<15} ({attr:+.4f})")

def main():
    model_path = "models/bert_finetuned_rjokes/"
    print(f"Loading model from {model_path}...")
    
    try:
        # Use MPS if available
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Sample jokes to analyze
    samples = [
        # Classic simple pun
        "What do you call a fake noodle? An Impasta.",
        
        # A likely "dad joke" or common repost structure
        "Why did the chicken cross the road? To get to the other side.",
        
        # A longer, narrative joke (setup + punchline)
        "My wife told me to stop impersonating a flamingo. I had to put my foot down.",
        
        # Meta-reddit humor (often scores high or very low depending on timing)
        "I finally read the dictionary. It turns out the zebra did it.",
        
        # Political/Topical (sensitive topics often behave differently)
        "What is the difference between a snowman and a snowwoman? Snowballs."
    ]

    print("Analyzing sample jokes...")
    for text in samples:
        print("-" * 50)
        print(f"Joke: {text}")
        tokens, attrs, score = compute_saliency(model, tokenizer, text)
        print_analysis(tokens, attrs, score)

if __name__ == "__main__":
    main()
