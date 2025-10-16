from logging.config import dictConfig

def setup_logging(
        level: str = "INFO",
        mute_chembl: bool = True
) -> None:
    """
    Configure logging for the application with suppression of noisy external libraries.
    Args:
        level: Logging level for the application ("DEBUG", "INFO", "WARNING", "ERROR")
        mute_chembl: If True, suppress verbose output from ChEMBL
    """
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "std"}
        },
        "root": {"level": level.upper(), "handlers": ["console"]},
        "loggers": {
            "chembl_webresource_client": {"level": "ERROR"},
        } if mute_chembl else {}
    })
