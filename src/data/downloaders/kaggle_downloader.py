import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KaggleDownloader:
    """
    Klasa odpowiedzialna za pobieranie bazowych zbiorów danych z Kaggle.
    """
    
    DATASETS = {
        "fakenewsnet": "mdepak/fakenewsnet",
        "pheme_dataset": "nicolemichelle/pheme-dataset-for-rumour-detection"
    }

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self._prepare_directories()
        
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Poprawna autoryzacja Kaggle API.")
        except OSError as e:
            logger.error("Błąd autoryzacji.")
            raise e

    def _prepare_directories(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Katalog docelowy ustawiony na: {self.raw_data_dir}")

    def download_dataset(self, dataset_key: str, unzip: bool = True) -> None:
        if dataset_key not in self.DATASETS:
            logger.error(f"Zbior '{dataset_key}' nie znajduje sie w slowniku DATASETS.")
            return

        dataset_slug = self.DATASETS[dataset_key]
        target_path = self.raw_data_dir / dataset_key
        target_path.mkdir(exist_ok=True)

        logger.info(f"Rozpoczynam pobieranie: {dataset_slug}...")
        
        try:
            self.api.dataset_download_files(
                dataset=dataset_slug, 
                path=str(target_path), 
                unzip=unzip
            )
            logger.info(f"Pobieranie '{dataset_key}' zakonczone sukcesem.")
            
        except Exception as e:
            logger.error(f"Bład podczas pobierania {dataset_key}: {e}")

    def download_all(self) -> None:
        logger.info("Rozpoczęto pobieranie danych...")
        for key in self.DATASETS.keys():
            self.download_dataset(key)
        logger.info("Pobieranie zakończone.")

if __name__ == "__main__":
    downloader = KaggleDownloader()
    downloader.download_all()