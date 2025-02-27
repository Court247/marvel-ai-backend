from pydantic import BaseModel, Field
from typing import List, Optional
import os
from app.services.logger import setup_logger
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)

    with open(absolute_file_path, 'r') as file:
        return file.read()
    
class SlideGenerator:
    def __init__(self, args=None, vectorstore_class=Chroma, prompt=None, embedding_model=None, model=None, parser=None, verbose=False):
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.5-flash"),
            "embedding_model": GoogleGenerativeAIEmbeddings(model='models/embedding-001'),
            "parser": JsonOutputParser(pydantic_object=SlidePresentation),
            "prompt": read_text_file("prompt/slide_generator_prompt.txt"),
            "vectorstore_class": Chroma
        }

        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        self.embedding_model = embedding_model or default_config["embedding_model"]

        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.vectorstore, self.retriever, self.runner = None, None, None
        self.args = args
        self.verbose = verbose

        if vectorstore_class is None: raise ValueError("Vectorstore must be provided")
       

        # # Return the chain
        # prompt = PromptTemplate(
        #     template=self.prompt,
        #     input_variables=["attribute_collection"],
        #     partial_variables={"format_instructions": self.parser.get_format_instructions()}
        # )

        # if self.runner is None:
        #     logger.info(f"Creating vectorstore from {len(documents)} documents") if self.verbose else None
        #     self.vectorstore = self.vectorstore_class.from_documents(documents, self.embedding_model)
        #     logger.info(f"Vectorstore created") if self.verbose else None

        #     self.retriever = self.vectorstore.as_retriever()
        #     logger.info(f"Retriever created successfully") if self.verbose else None

        #     self.runner = RunnableParallel(
        #         {"context": self.retriever,
        #         "attribute_collection": RunnablePassthrough()
        #         }
        #     )

        # chain = self.runner | prompt | self.model | self.parser

        # logger.info(f"Chain compilation complete")

        # return chain
    
    def compile_with_context(self):
        # Return the chain
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["instructional_level", "topic", "slides_titles"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.model | self.parser

        logger.info(f"Chain compilation complete")

        return chain

    def generate_slides(self):
        logger.info(f"Creating the Outlines for the Presentation") 

        chain = self.compile_with_context()
        

        input_parameters = {
            "instructional_level": self.args.instructional_level,
            "topic": self.args.topic,
            "slides_titles": self.args.slides_titles,
            "lang": self.args.lang
        }
        logger.info(f"Input parameters: {input_parameters}")

        response = chain.invoke(input_parameters)

        logger.info(f"Generated response: {response}")


        return response

class Slide(BaseModel):
    title: str = Field(..., description="The title of the slide")
    template: str = Field(..., description="The slide template type: sectionHeader, titleAndBody, titleAndBullets, twoColumn")
    content: Optional[str] = Field(None, description="For sectionHeader slides, a brief description")
    # subtitle: Optional[str] = Field(None, description="For sectionHeader slides, an optional subtitle")
    # body: Optional[str] = Field(None, description="For titleAndBody slides, a paragraph of text")
    # bullets: Optional[List[str]] = Field(None, description="For titleAndBullets slides, a list of bullet points")
    # leftColumn: Optional[str] = Field(None, description="For twoColumn slides, the left column content")
    # rightColumn: Optional[str] = Field(None, description="For twoColumn slides, the right column content")

class SlidePresentation(BaseModel):
    slides: List[Slide] = Field(..., description="The complete set of slides in the presentation")