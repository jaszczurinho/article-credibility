import math
import nltk
import numpy as np
from scipy.stats import entropy
from collections import Counter
from transformers import pipeline, AutoTokenizer
import logging
from functools import lru_cache

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

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

nltk.download("punkt", quiet=True)

@lru_cache(maxsize=1)
def load_emotion_model():
    logger.info("Ładuję model językowy: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    emotion_classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
    )
    logger.info("Model językowy załadowany")
    return tokenizer, emotion_classifier


def split_long_sentence(tokenizer, token_ids, max_content_tokens):
    """Split pre-computed token_ids into chunks to avoid double encoding."""
    for start in range(0, len(token_ids), max_content_tokens):
        yield tokenizer.decode(token_ids[start:start + max_content_tokens], skip_special_tokens=True)


def calculate_article_features(text, batch_size=16):
    tokenizer, emotion_classifier = load_emotion_model()
    max_content_tokens = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False)

    # Inicjalizacja domyślnego słownika wyników; na wypadek braku tekstu
    results_dict = {
        "emotion_entropy": 0.0,
        "chunk_count": 0
    }
    for emotion in EMOTION_CLASSES:
        results_dict[f"first_chunk_prob_{emotion}"] = 0.0

    text = "" if (text is None or (isinstance(text, float) and math.isnan(text))) else str(text)
    if not text.strip():
        return results_dict

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_len = len(token_ids)

        if sentence_len > max_content_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            chunks.extend(split_long_sentence(tokenizer, token_ids, max_content_tokens))
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
        return results_dict

    # Batch all chunks in a single forward pass 
    all_predictions = emotion_classifier(
        chunks,
        batch_size=batch_size,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    for pred in all_predictions[0]:
        results_dict[f"first_chunk_prob_{pred['label']}"] = pred['score']

    prob_matrix = np.zeros((len(chunks), len(EMOTION_CLASSES)))
    emotion_index = {e: i for i, e in enumerate(EMOTION_CLASSES)}

    for chunk_idx, preds in enumerate(all_predictions):
        for pred in preds:
            col = emotion_index.get(pred['label'])
            if col is not None:
                prob_matrix[chunk_idx, col] = pred['score']

    mean_probs = prob_matrix.mean(axis=0)
    max_entropy = np.log2(len(EMOTION_CLASSES))
    shannon_entropy = entropy(mean_probs, base=2)
    results_dict["emotion_entropy"] = float(shannon_entropy / max_entropy) if max_entropy > 0 else 0.0
    results_dict["chunk_count"] = len(chunks)

    return results_dict