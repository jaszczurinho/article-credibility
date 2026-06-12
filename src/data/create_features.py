import pandas as pd
import spacy

# Wyznaczamy encje, czyli osoby, organizacje, miejsca i liczby, a następnie obliczamy ich gęstość względem liczby słów w tekście
def calculate_entity_densities(text: pd.Series, nlp_model):
    text = str(text)
    doc = nlp_model(text)
    word_count = len(text.split()) if len(text.split()) > 0 else 1
    
    entities_count = {'PERSON': 0, 'ORG': 0, 'GPE': 0, 'CARDINAL': 0}
    
    for ent in doc.ents:
        if ent.label_ in entities_count:
            entities_count[ent.label_] += 1
            
    return pd.Series({
        'person_density': entities_count['PERSON'] / word_count,
        'org_density': entities_count['ORG'] / word_count,
        'gpe_density': entities_count['GPE'] / word_count,
        'cardinal_density': entities_count['CARDINAL'] / word_count
    })

# Wyznaczamy gęstość zaimków osobowych, przymiotników i spójników
def pos_features(text: pd.Series, nlp_model):
    text = str(text)
    doc = nlp_model(text)
    
    valid_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    word_count = len(valid_tokens) if len(valid_tokens) > 0 else 1
    
    counts = {
        'p1': 0, 'p2': 0, 'p3': 0, 
        'adj': 0, 'conj': 0
    }
    
    for token in valid_tokens:
        # 1. Zaimki osobowe
        if token.pos_ == 'PRON':
            person = token.morph.get("Person")
            if "1" in person: counts['p1'] += 1   # pierwsza osoba
            elif "2" in person: counts['p2'] += 1 # druga osoba
            elif "3" in person: counts['p3'] += 1 # trzecia osoba
        
        # 2. Przymiotniki
        elif token.pos_ == 'ADJ':
            counts['adj'] += 1
            
        # 3. Spójniki
        elif token.pos_ in ['CCONJ', 'SCONJ']:
            counts['conj'] += 1

    return pd.Series({
        'pronoun_1st_density': counts['p1'] / word_count,
        'pronoun_2nd_density': counts['p2'] / word_count,
        'pronoun_3rd_density': counts['p3'] / word_count,
        'adj_density': counts['adj'] / word_count,
        'conj_density': counts['conj'] / word_count
    })

# wyznaczamy gęstość cytowań
def quote_density(text: pd.Series):
    text = str(text)

    tokens = text.split()
    word_count = len(tokens)

    if word_count == 0:
        return 0

    quote_chars = ['"', "'", "“", "”", "‘", "’", "«", "»"]
    quote_count = sum(text.count(char) for char in quote_chars)

    return quote_count / word_count


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    data = pd.read_csv("data/processed/final_dataset.csv")
    
    print("Wyznaczamy POS densities...")
    # POS densities
    stylometryczne_cechy = data['text'].apply(lambda x: pos_features(x, nlp))
    data = pd.concat([data, stylometryczne_cechy], axis=1)
    
    print("Wyznaczamy Quote density...")
    # Quote density
    data['quote_density'] = data['text'].apply(quote_density)
    
    print("Wyznaczamy Entity densities...")
    # Entity densities
    entity_densities = data['text'].apply(lambda x: calculate_entity_densities(x, nlp))
    data = pd.concat([data, entity_densities], axis=1)
    
    data.to_csv("data/processed/final_dataset_with_features.csv", index=False)