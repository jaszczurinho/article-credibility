import logging
from pathlib import Path
from .downloaders.reliable_articles_scraper import main as newsapi_main
from .downloaders.kaggle_downloader import KaggleDownloader
from .downloaders.download_from_github import clone_github_repo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataOrchestrator:
    
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_newsapi_data(self) -> None:
        logger.info("Downloading NewsAPI data...")
        try:
            newsapi_main()
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
    
    def download_kaggle_datasets(self) -> None:
        logger.info("Downloading Kaggle datasets...")
        try:
            downloader = KaggleDownloader(str(self.raw_data_dir))
            downloader.download_all()
        except Exception as e:
            logger.error(f"Kaggle error: {e}")
    
    def download_github_repos(self) -> None:
        logger.info("Downloading GitHub repositories...")
        try:
            clone_github_repo(
                "https://github.com/cuilimeng/CoAID.git",
                str(self.raw_data_dir / "coaid")
            )
        except Exception as e:
            logger.error(f"GitHub error: {e}")
    
    def run_all(self) -> None:
        self.download_newsapi_data()
        self.download_kaggle_datasets()
        self.download_github_repos()


def main():
    orchestrator = DataOrchestrator()
    orchestrator.run_all()


if __name__ == "__main__":
    main()
