import os
import re
import time
import logging
import pandas as pd
from newsapi import NewsApiClient
from newspaper import Article, Config
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RELIABLE_DOMAINS = "apnews.com,theguardian.com,npr.org,politico.com,axios.com,reuters.com,cnbc.com,cbsnews.com"
UNRELIABLE_DOMAINS = "thegatewaypundit.com,wnd.com,americanthinker.com,westernjournal.com,globalresearch.ca,activistpost.com,lewrockwell.com,zerohedge.com,breitbart.com,dailymail.co.uk,nypost.com"

QUERIES = ["climate change", "artificial intelligence", "war", "cybersecurity", "health", "economy", "education", "pandemic", "politics"]

def clean_article_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def get_newspaper_config():
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    config.request_timeout = 10
    config.memoize_articles = False
    return config

def fetch_newsapi_dataset(api_key: str, domains: str, label: int, source_category: str, articles_per_query: int) -> list[dict]:
    newsapi = NewsApiClient(api_key=api_key)
    scraped_data = []
    newspaper_cfg = get_newspaper_config()
    
    seen_urls = set() # Zbiór do śledzenia pobranych linków
    
    for query in QUERIES:
        logger.info(f"--- [LABEL {label}] Wyszukiwanie dla query: '{query}' ---")
        
        try:
            response = newsapi.get_everything(
                q=query,
                domains=domains,
                language='en',
                sort_by='popularity',
                page_size=articles_per_query
            )
        except Exception as e:
            logger.error(f"Błąd NewsAPI dla query '{query}': {e}")
            continue
            
        articles = response.get('articles', [])
        logger.info(f"Zwrócono {len(articles)} artykułów dla query '{query}'.")
        
        for art in articles:
            url = art.get('url')
            publisher_name = art.get('source', {}).get('name', 'Unknown')
            
            if not url or "google.com" in url:
                continue
                
            # Sprawdzanie czy URL już istnieje w zbiorze
            if url in seen_urls:
                continue 
                
            seen_urls.add(url) 
                
            try:
                article = Article(url, config=newspaper_cfg)
                article.download()
                article.parse()
                
                full_text = clean_article_text(article.text)
                
                # Filtr jakościowy
                if len(full_text) > 300:
                    scraped_data.append({
                        "text": full_text,
                        "label": label,
                        "source_category": source_category,
                        "publisher": publisher_name
                    })
                    logger.info(f"[ZAPISANO] -> {article.title[:45]}... ({publisher_name})")
                    
            except Exception:
                continue
                
            time.sleep(0.5)
            
        time.sleep(1.0)
            
    return scraped_data

def main():
    api_key = os.getenv("NEWSAPI_KEY")
    
    if not api_key:
        logger.error("Brak klucza NEWSAPI_KEY w środowisku.")
        return

    ARTICLES_PER_QUERY = 100 
    output_dir = Path("data/raw/newsapi_ai_era")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== POBIERANIE RZETELNYCH ARTYKUŁÓW ===")
    reliable_results = fetch_newsapi_dataset(
        api_key=api_key,
        domains=RELIABLE_DOMAINS,
        label=0,
        source_category="news_api",
        articles_per_query=ARTICLES_PER_QUERY
    )
    
    if reliable_results:
        df_reliable = pd.DataFrame(reliable_results)
        reliable_path = output_dir / "reliable_news_api.csv"
        df_reliable.to_csv(reliable_path, index=False, encoding='utf-8')
        logger.info(f"Zapisano {len(df_reliable)} rzetelnych artykułów w {reliable_path}")

    logger.info("=== POBIERANIE NIERZETELNYCH ARTYKUŁÓW ===")
    unreliable_results = fetch_newsapi_dataset(
        api_key=api_key,
        domains=UNRELIABLE_DOMAINS,
        label=1,
        source_category="news_api",
        articles_per_query=ARTICLES_PER_QUERY
    )
    
    if unreliable_results:
        df_unreliable = pd.DataFrame(unreliable_results)
        unreliable_path = output_dir / "unreliable_news_api.csv"
        df_unreliable.to_csv(unreliable_path, index=False, encoding='utf-8')
        logger.info(f"Zapisano {len(df_unreliable)} nierzetelnych artykułów w {unreliable_path}")

if __name__ == "__main__":
    main()
