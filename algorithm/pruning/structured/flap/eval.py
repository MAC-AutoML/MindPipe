import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    """
    Evaluate perplexity (ppl) on a specified model and tokenizer.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the test dataset.
    """
    # Set dataset
    dataset = "wikitext2"   # Dataset consisting of extracted sentences from Wikipedia articles

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate perplexity in no grad context to avoid updating the model
    with torch.no_grad():
        # Perplexity measures how well the probability distribution predicted by the model aligns with the actual distribution of the words. Lower perplexity is better.
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    # Get input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        
        # Forward pass through the model
        lm_logits = model(inputs).logits    

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()    # Example: [cat, sat, on, ???] -> [cat, sat, on]
        shift_labels = inputs[:, 1:]    # Example: [The, cat, sat, on] -> [cat, sat, on]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)    # nll = loss * sequence_length * batch_size

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))    # ppl = exp(∑(nlls) / (num_samples * sequence_length))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Note: 
# 1. Perplexity (ppl) is a measure of how well a probability model predicts a sample. 
# 2. Lower perplexity indicates better performance of the model.
# 3. In this script, the perplexity of a language model is evaluated using a specific dataset ('wikitext2').
