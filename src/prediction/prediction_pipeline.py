import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

EMB_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
MODELS_DIR     = Path('./models')
INPUT_DIR      = Path('./input')
OUTPUT_DIR     = Path('./output')


def load_emb_model():
    print(f'Loading embedding model: {EMB_MODEL_NAME}')
    return BGEM3FlagModel(EMB_MODEL_NAME, use_fp16=True)


def encode_texts(texts: list[str], emb_model: BGEM3FlagModel):
    print(f'Encoding {len(texts)} texts...')
    output = emb_model.encode(texts, batch_size=16)
    return np.array(output['dense_vecs'])


def load_input(file_arg: str | None) -> pd.DataFrame:
    if file_arg:
        path = Path(file_arg)
    else:
        csv_files = list(INPUT_DIR.glob('*.csv'))
        assert len(csv_files) > 0, f'No CSV files found in {INPUT_DIR}/'
        assert len(csv_files) == 1, (
            f'There are multiple CSV files in {INPUT_DIR}/: '
            f'{[f.name for f in csv_files]}. Specify the file with --file.'
        )
        path = csv_files[0]

    print(f'Loading data from: {path}')
    df = pd.read_csv(path)
    assert 'text' in df.columns, f'File {path} must contain a "text" column with article texts.'
    return df


def predict(texts: list[str], model_name: str = 'stacking') -> pd.DataFrame:
    model_path = MODELS_DIR / f'{model_name}_model.joblib'
    assert model_path.exists(), f'File not found: {model_path}'

    model     = joblib.load(model_path)
    emb_model = load_emb_model()
    X_emb     = encode_texts(texts, emb_model)

    preds  = model.predict(X_emb)
    probas = model.predict_proba(X_emb)[:, 1] if hasattr(model, 'predict_proba') else None

    results = pd.DataFrame({
        'text': texts,
        'prediction': preds,
        'label': pd.Series(preds),
        'score': probas
    })
    if probas is not None:
        results['confidence'] = probas.round(3)

    return results

def save_results(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / 'predictions.csv'
    results.to_csv(out_path, index=False)
    print(f'Saved to: {out_path}')

def main():
    parser = argparse.ArgumentParser(description='Article credibility prediction pipeline')

    parser.add_argument('--file', type=str, default=None,
                        help='Path to the CSV file (default: first file in ./input/)')
    parser.add_argument('--model', type=str, default='stacking',
                    help='Model name without extension')
    
    args = parser.parse_args()
    df      = load_input(args.file)
    results = predict(df['text'].tolist(), args.model)
    save_results(results)


if __name__ == '__main__':
    main()