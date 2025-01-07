import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
import argparse
import re

# Step 1: Preprocess the text
def preprocess_text(file_path):
    with open(file_path, 'r') as f:
        text = f.read()  # Read up to 11000 characters

    # Remove newlines
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')

    # Remove all non-alphabetic characters (keeping only а-я and А-Я)
    text = re.sub(r'[^а-яА-Я\-\s]', '', text)  # \s keeps spaces, remove if you don't want spaces

    # Convert to lowercase
    text = text.lower()

    # Split into tokens (words)
    text = text.split()        
        
    print('step 1: OK')
    return text

# Step 2: Create vocabulary and word-to-index mapping
def create_vocab(text, vocab_size=30000):
    word_counts = Counter(text)
    vocab = word_counts.most_common(vocab_size - 1)  # Keep most frequent words
    vocab = [word for word, _ in vocab]
    vocab.append('<UNK>')  # Add unknown token
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    print("Vocab:", vocab[:10])
    return word_to_idx, vocab

# Step 3: Generate CBOW training data
def generate_cbow_data(text, word_to_idx, window_size=2):
    data = []
    for i in range(window_size, len(text) - window_size):
        target = word_to_idx.get(text[i], word_to_idx['<UNK>'])
        context = [
            word_to_idx.get(text[i + j], word_to_idx['<UNK>'])
            for j in range(-window_size, window_size + 1)
            if j != 0
        ]
        data.append((context, target))
    return data

# Step 4: Define a PyTorch Dataset
class CBOWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Step 5: Define the CBOW model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=1)  # Average context embeddings
        out = self.linear(embeds)
        return out

# Step 6: Train the CBOW model
def train_cbow(file_path, embedding_dim=100, window_size=2, batch_size=32, epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess text and create vocabulary
    text = preprocess_text(file_path)
    word_to_idx, vocab = create_vocab(text)
    vocab_size = len(vocab)

    # Generate CBOW data
    data = generate_cbow_data(text, word_to_idx, window_size)
    dataset = CBOWDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = CBOW(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return model, word_to_idx, vocab


# Add this after the training code

# Step 8: Test the trained CBOW model
def test_cbow(model, word_to_idx, vocab, test_words=['офицер']):
    # Retrieve the embedding layer weights
    embeddings = model.embeddings.weight.data

    # Function to find the most similar words
    def find_most_similar(word, top_k=5):
        if word not in word_to_idx:
            print(f"Word '{word}' not in vocabulary!")
            return
        word_idx = word_to_idx[word]
        word_embedding = embeddings[word_idx]

        # Compute cosine similarity between the word and all other words
        cosine_sim = torch.nn.CosineSimilarity(dim=1)
        similarities = cosine_sim(word_embedding.unsqueeze(0), embeddings)

        # Get the top-k most similar words
        top_indices = similarities.argsort(descending=True)[1:top_k + 1]  # Exclude the word itself
        most_similar = [(vocab[idx], similarities[idx].item()) for idx in top_indices]
        return most_similar

    # Function to predict a target word given a context
    def predict_target_word(context_words):
        context_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in context_words]
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(context_tensor.to('cuda'))
            predicted_index = torch.argmax(output, dim=1).item()
        return vocab[predicted_index]


    # Test 2: Find most similar words
    for test_word in test_words:
        print(f"\nMost similar words to '{test_word}':")
        most_similar = find_most_similar(test_word, top_k=5)
        for word, similarity in most_similar:
            print(f"{word}: {similarity:.4f}")

    # Test 3: Predict target word given a context
    context = test_words
    predicted_word = predict_target_word(context)
    print(f"\nGiven the context '{' '.join(context)}', the predicted target word is: '{predicted_word}'")
    
    
import pickle  # Add this import at the top of the file

# Step 10: Save the model, word_to_idx, and vocab using pickle
def save_artifacts(model, word_to_idx, vocab, file_prefix="cbow"):
    # Save the model
    model_path = f"{file_prefix}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    # Save the word_to_idx mapping
    word_to_idx_path = f"{file_prefix}_word_to_idx.pkl"
    with open(word_to_idx_path, "wb") as f:
        pickle.dump(word_to_idx, f)
    print(f"word_to_idx saved to {word_to_idx_path}")

    # Save the vocabulary
    vocab_path = f"{file_prefix}_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")

# Step 11: Load the model, word_to_idx, and vocab from pickle files
def load_artifacts(file_prefix="cbow"):
    # Load the model
    model_path = f"{file_prefix}_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")

    # Load the word_to_idx mapping
    word_to_idx_path = f"{file_prefix}_word_to_idx.pkl"
    with open(word_to_idx_path, "rb") as f:
        word_to_idx = pickle.load(f)
    print(f"word_to_idx loaded from {word_to_idx_path}")

    # Load the vocabulary
    vocab_path = f"{file_prefix}_vocab.pkl"
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {vocab_path}")

    return model, word_to_idx, vocab    
    
# Step 7: Run the training
if __name__ == "__main__":
    file_path = "voina-i-mir.txt"  # Replace with your text file path

    parser = argparse.ArgumentParser(description="Script to handle --test and --train modes.")
    
    # Add --test argument
    parser.add_argument('--test', nargs='*', help='Enable test mode')
    
    # Add --train argument
    parser.add_argument('--train', action='store_true', help='Enable train mode')
    parser.add_argument('--epochs', type=int, default=10, help='# epochs to train')
    
    
    # Parse the arguments
    args = parser.parse_args()

    if args.train:
        print("Train mode is enabled")
        # Add your train mode logic here
        model, word_to_idx, vocab = train_cbow(file_path, epochs=args.epochs)
        # Save model
        save_artifacts(model, word_to_idx, vocab, file_prefix="cbow")
    
    # Check which mode is enabled
    if args.test:
        print("Test mode is enabled")
        model, word_to_idx, vocab = load_artifacts()
        # Add your test mode logic here
        print("words:", args.test)
        # Step 9: Run the testing
        test_cbow(model, word_to_idx, vocab, args.test)    
    
    
