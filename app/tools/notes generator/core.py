#Executor function goes here with input parameters matching the tool's functionality including type hints

from app.services.logger import setup_logger


logger = setup_logger(__name__)

SUPPORTED_FILE_TYPES = [
    'pdf', 'txt', 'docx'
    ]

