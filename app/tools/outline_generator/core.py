from app.utils.document_loaders import get_docs
from app.services.schemas import OutlineGeneratorInput
from app.services.logger import setup_logger
from app.api.error_utilities import LoaderError, ToolExecutorError
from app.tools.outline_generator.tools import OutlineGenerator
logger = setup_logger()

def executor(
             n_slides: int,
             topic: str,
             instructional_level: str,
             file_upload_url: str,
             file_upload_type: str,
             lang: str, 
             verbose=False):

    try:
        if (not (n_slides and topic and instructional_level)):
            logger.info(f"Missing required inputs.")
         
        if(n_slides and topic and instructional_level):
            logger.info(f"Generating slide outlines. from {topic} for {instructional_level} level")
        if(file_upload_url and file_upload_type):
            logger.info(f"Fetching documents from {file_upload_url} of type {file_upload_type}")
        docs = None

        def fetch_docs(file_url, file_type):
            return get_docs(file_url, file_type, True) if file_url and file_type else None

        docs = fetch_docs(file_type=file_upload_type, file_url=file_upload_url)
        


        presentation_generator_args = OutlineGeneratorInput(
            instructional_level=instructional_level,
            n_slides=n_slides,
            topic=topic,
            file_upload_url=file_upload_url,
            file_upload_type=file_upload_type,
            lang=lang
        )
        print("docs", docs)
        output = OutlineGenerator(args=presentation_generator_args, verbose=verbose).generate_outline(docs)

        logger.info(f"Presentation generated successfully")

    except LoaderError as e:
        error_message = e
        logger.error(f"Error in Presentation Generator Pipeline -> {error_message}")
        raise ToolExecutorError(error_message)

    except Exception as e:
        error_message = f"Error in executor: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    return output