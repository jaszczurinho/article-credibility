import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clone_github_repo(repo_url: str, target_dir: str) -> None:
    """
    Uniwersalna funkcja do klonowania dowolnego repozytorium GitHub 
    do wskazanego katalogu w projekcie.
    
    Argumenty:
        repo_url (str): Pełny adres URL repozytorium.
        target_dir (str): Ścieżka docelowa, gdzie mają wylądować dane.
    """
    target_path = Path(target_dir)

    # Zabezpieczenie przed ponownym klonowaniem 
    if target_path.exists() and any(target_path.iterdir()):
        logger.info(f"Katalog {target_path} już istnieje i zawiera dane. Pomijam pobieranie dla: {repo_url}")
        return

    # Tworzenie struktury katalogów nadrzędnych, jeśli nie istnieją
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Rozpoczynam klonowanie repozytorium z {repo_url} do {target_path}...")

    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_path)], 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Repozytorium zostało pomyślnie pobrane do: {target_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Błąd Git podczas klonowania repozytorium {repo_url}: {e.stderr}")

if __name__ == "__main__":
    clone_github_repo(
        repo_url="https://github.com/cuilimeng/CoAID.git",
        target_dir="data/raw/coaid"
    )