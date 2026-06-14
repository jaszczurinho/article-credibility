import pandas as pd
import spacy
import logging
from tqdm.auto import tqdm

from emotion_extraction import calculate_article_features

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_dataframe_in_parallel(df: pd.DataFrame, text_column: str, nlp_model) -> pd.DataFrame:
    texts = df[text_column].astype(str).tolist()
    total_docs = len(texts)
    results = []
    
    quote_chars = ['"', "'", "“", "”", "‘", "’", "«", "»"]
    
    logger.info(f"Rozpoczynam analizę tekstów z użyciem spaCy...")
    
    for i, doc in enumerate(nlp_model.pipe(texts, n_process=-1, batch_size=50), 1):
        
        if i % 50 == 0 or i == total_docs:
            logger.info(f"Przetworzono {i} tekstów")
            
        word_count_ner = len(doc.text.split()) if len(doc.text.split()) > 0 else 1
        
        quote_count = sum(doc.text.count(char) for char in quote_chars)
        quote_density = quote_count / word_count_ner
        
        valid_tokens = [t for t in doc if not t.is_space and not t.is_punct]
        word_count_pos = len(valid_tokens) if len(valid_tokens) > 0 else 1
        
        # --- ANALIZA NER ---
        entities_count = {'PERSON': 0, 'ORG': 0, 'GPE': 0, 'CARDINAL': 0}
        for ent in doc.ents:
            if ent.label_ in entities_count:
                entities_count[ent.label_] += 1
                
        # --- ANALIZA POS ---
        pos_counts = {'p1': 0, 'p2': 0, 'p3': 0, 'adj': 0, 'conj': 0}
        for token in valid_tokens:
            if token.pos_ == 'PRON':
                person = token.morph.get("Person")
                if "1" in person: pos_counts['p1'] += 1
                elif "2" in person: pos_counts['p2'] += 1
                elif "3" in person: pos_counts['p3'] += 1
            elif token.pos_ == 'ADJ':
                pos_counts['adj'] += 1
            elif token.pos_ in ['CCONJ', 'SCONJ']:
                pos_counts['conj'] += 1
                
        results.append({
            'person_density': entities_count['PERSON'] / word_count_ner,
            'org_density': entities_count['ORG'] / word_count_ner,
            'gpe_density': entities_count['GPE'] / word_count_ner,
            'cardinal_density': entities_count['CARDINAL'] / word_count_ner,
            'pronoun_1st_density': pos_counts['p1'] / word_count_pos,
            'pronoun_2nd_density': pos_counts['p2'] / word_count_pos,
            'pronoun_3rd_density': pos_counts['p3'] / word_count_pos,
            'adj_density': pos_counts['adj'] / word_count_pos,
            'conj_density': pos_counts['conj'] / word_count_pos,
            'quote_density': quote_density
        })
        
    features_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), features_df], axis=1)

if __name__ == '__main__':
    logger.info("Wczytywanie pliku CSV z danymi...")
    data = pd.read_csv("data/processed/final_dataset.csv")
    
    logger.info("Ładowanie modelu językowego en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")

    tqdm.pandas(desc="Tworzenie cech do analizy emocji tekstów")
    emotion_features_series = data["text"].progress_apply(calculate_article_features)
    emotion_features_df = pd.DataFrame(emotion_features_series.tolist())

    logger.info("Tworzenie cech z uzyciem spaCy...")
    df_final = process_dataframe_in_parallel(data, 'text', nlp)
    
    df_final = pd.concat([df_final.reset_index(drop=True), emotion_features_df.reset_index(drop=True)], axis=1)
    
    output_path = "data/processed/final_dataset_with_features.csv"

    df_final.to_csv(output_path, index=False)
    logger.info(f"Plik zapisany jako: {output_path}")