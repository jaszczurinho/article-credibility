import nltk
import numpy as np
from scipy.stats import entropy
from collections import Counter
import pandas as pd
from transformers import pipeline, AutoTokenizer
import logging
from functools import lru_cache
from tqdm.auto import tqdm

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

nltk.download("punkt", quiet=True)  # Do dzielenia tekstu na zdania


@lru_cache(maxsize=1)
def load_emotion_model():
    logger.info("Ładuję model emocji: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    emotion_classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
    )
    logger.info("Model emocji zaladowany")
    return tokenizer, emotion_classifier


def split_long_sentence(tokenizer, sentence, max_content_tokens):
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    for start in range(0, len(token_ids), max_content_tokens):
        yield tokenizer.decode(token_ids[start:start + max_content_tokens], skip_special_tokens=True)


def calculate_article_entropy(text, max_tokens=512):
    tokenizer, emotion_classifier = load_emotion_model()
    model_max_tokens = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False)
    max_content_tokens = min(max_tokens, model_max_tokens)

    text = "" if pd.isna(text) else str(text)
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    # Step 1: Maximize chunks based on max_tokens limit
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_len = len(sentence_tokens)

        if sentence_len > max_content_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0

            chunks.extend(split_long_sentence(tokenizer, sentence, max_content_tokens))
            continue

        if current_token_count + sentence_len > max_content_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        current_chunk.append(sentence)
        current_token_count += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks:
        return 0

    # Step 2: Classify emotions for each chunk
    chunk_emotions = []
    for chunk in chunks:
        results = emotion_classifier(
            chunk,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )[0]
        # Get the label with the highest score
        top_emotion = max(results, key=lambda x: x['score'])['label']
        chunk_emotions.append(top_emotion)

    # Step 3: Calculate entropy
    counts = Counter(chunk_emotions)
    
    # Normalize by max possible entropy to keep it 0-1
    num_emotions = 7  # j-hartmann model has 7 classes
    max_entropy = np.log2(num_emotions)
    
    shannon_entropy = entropy(list(counts.values()), base=2)
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


if __name__ == "__main__":
    logger.info("Wczytywanie danych...")
    data = pd.read_csv("data/processed/final_dataset_with_features.csv")[:500]

    logger.info("Wyznaczanie emotion_entropy dla %s tekstów", len(data))

    tqdm.pandas(desc="Liczenie emotion_entropy")
    data["emotion_entropy"] = data["text"].progress_apply(calculate_article_entropy)

    logger.info("Wyniki zapisane")

    data.to_csv("data/processed/final_dataset_with_emotion_entropy.csv", index=False)

