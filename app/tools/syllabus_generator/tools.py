from pydantic import BaseModel, Field
from typing import List, Dict
from app.services.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel  # (still used for the parallel branch)
from app.services.schemas import SyllabusGeneratorArgsModel
from fastapi import HTTPException

logger = setup_logger(__name__)

class SyllabusRequestArgs:
    def __init__(self, syllabus_generator_args: SyllabusGeneratorArgsModel, summary: str):
        self._grade_level = syllabus_generator_args.grade_level
        self._subject = syllabus_generator_args.subject
        self._course_description = syllabus_generator_args.course_description
        self._objectives = syllabus_generator_args.objectives
        self._required_materials = syllabus_generator_args.required_materials
        self._grading_policy = syllabus_generator_args.grading_policy
        self._policies_expectations = syllabus_generator_args.policies_expectations
        self._course_outline = syllabus_generator_args.course_outline
        self._additional_notes = syllabus_generator_args.additional_notes
        self._lang = syllabus_generator_args.lang
        self._summary = summary

    def to_dict(self) -> dict:
        return {
            "grade_level": self._grade_level,
            "subject": self._subject,
            "course_description": self._course_description,
            "objectives": self._objectives,
            "required_materials": self._required_materials,
            "grading_policy": self._grading_policy,
            "policies_expectations": self._policies_expectations,
            "course_outline": self._course_outline,
            "additional_notes": self._additional_notes,
            "lang": self._lang,
            "summary": self._summary,
        }

class SyllabusGeneratorPipeline:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model = GoogleGenerativeAI(model="gemini-1.5-pro")
        self.parsers = {
            "course_information": JsonOutputParser(pydantic_object=CourseInformation),
            "course_description_objectives": JsonOutputParser(pydantic_object=CourseDescriptionObjectives),
            "course_content": JsonOutputParser(pydantic_object=CourseContentItem),
            "policies_procedures": JsonOutputParser(pydantic_object=PoliciesProcedures),
            "assessment_grading_criteria": JsonOutputParser(pydantic_object=AssessmentGradingCriteria),
            "learning_resources": JsonOutputParser(pydantic_object=LearningResource),
            "course_schedule": JsonOutputParser(pydantic_object=CourseScheduleItem),
        }

    # ===== NEW METHOD: compile_sequential() to build a hybrid pipeline =====
    def compile_sequential(self):
        try:
            # --- Chain for course_information (sequential step 1) ---
            course_info_prompt = PromptTemplate(
                template=(
                    "Generate a detailed and structured course information in {lang} based on:\n\n"
                    "Grade Level: {grade_level}\n"
                    "Subject: {subject}\n"
                    "Course Description: {course_description}\n"
                    "Summary: {summary}\n\n"
                    "Ensure the response is professional and comprehensive.\n{format_instructions}"
                ),
                input_variables=["grade_level", "subject", "course_description", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["course_information"].get_format_instructions()},
            )
            self.chain_course_information = course_info_prompt | self.model | self.parsers["course_information"]

            # --- Chain for course_description_objectives (sequential step 1 parallel branch) ---
            course_desc_obj_prompt = PromptTemplate(
                template=(
                    "Develop detailed course objectives and intended learning outcomes in {lang}:\n\n"
                    "Objectives: {objectives}\n"
                    "Summary: {summary}\n\n"
                    "Provide measurable goals and realistic expectations for students.\n{format_instructions}"
                ),
                input_variables=["objectives", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["course_description_objectives"].get_format_instructions()},
            )
            self.chain_course_description_objectives = course_desc_obj_prompt | self.model | self.parsers["course_description_objectives"]

            # --- Chain for course_content (sequential step 2, uses output from course_information) ---
            course_content_prompt = PromptTemplate(
                template=(
                    "Using the following course information, generate a detailed course content outline in {lang}:\n\n"
                    "Course Information: {course_information}\n"
                    "Course Outline: {course_outline}\n"
                    "Summary: {summary}\n\n"
                    "Include topics, time frames, and key learning points.\n{format_instructions}"
                ),
                input_variables=["course_information", "course_outline", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["course_content"].get_format_instructions()},
            )
            self.chain_course_content = course_content_prompt | self.model | self.parsers["course_content"]

            # --- Chain for policies_procedures (sequential step 3) ---
            policies_prompt = PromptTemplate(
                template=(
                    "Draft clear and professional course policies and procedures in {lang}:\n\n"
                    "Grading Policy: {grading_policy}\n"
                    "Class Policies and Expectations: {policies_expectations}\n"
                    "Summary: {summary}\n\n"
                    "Ensure all rules and expectations are outlined clearly.\n{format_instructions}"
                ),
                input_variables=["grading_policy", "policies_expectations", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["policies_procedures"].get_format_instructions()},
            )
            self.chain_policies_procedures = policies_prompt | self.model | self.parsers["policies_procedures"]

            # --- Parallel chains for assessment, learning resources, course schedule ---
            assessment_prompt = PromptTemplate(
                template=(
                    "Define assessment methods and grading criteria in {lang}:\n\n"
                    "Grading Policy: {grading_policy}\n"
                    "Summary: {summary}\n\n"
                    "Ensure that assessment methods and the grading scale are precise and easy to understand.\n{format_instructions}"
                ),
                input_variables=["grading_policy", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["assessment_grading_criteria"].get_format_instructions()},
            )
            self.chain_assessment_grading_criteria = assessment_prompt | self.model | self.parsers["assessment_grading_criteria"]

            learning_resources_prompt = PromptTemplate(
                template=(
                    "Generate a comprehensive list of recommended learning resources in {lang}:\n\n"
                    "Required Materials: {required_materials}\n"
                    "Summary: {summary}\n\n"
                    "Include titles, authors, and publication years of the materials.\n{format_instructions}"
                ),
                input_variables=["required_materials", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["learning_resources"].get_format_instructions()},
            )
            self.chain_learning_resources = learning_resources_prompt | self.model | self.parsers["learning_resources"]

            course_schedule_prompt = PromptTemplate(
                template=(
                    "Construct a detailed course schedule in {lang}:\n\n"
                    "Course Outline: {course_outline}\n"
                    "Summary: {summary}\n\n"
                    "Ensure the schedule includes dates, activities, and key topics.\n{format_instructions}"
                ),
                input_variables=["course_outline", "lang", "summary"],
                partial_variables={"format_instructions": self.parsers["course_schedule"].get_format_instructions()},
            )
            self.chain_course_schedule = course_schedule_prompt | self.model | self.parsers["course_schedule"]

            # Build a parallel pipeline for the chains that can be executed concurrently
            self.parallel_pipeline = RunnableParallel(branches={
                "assessment_grading_criteria": self.chain_assessment_grading_criteria,
                "learning_resources": self.chain_learning_resources,
                "course_schedule": self.chain_course_schedule,
            })

            if self.verbose:
                logger.info("Successfully compiled the hybrid sequential and parallel pipeline.")

        except Exception as e:
            logger.error(f"Failed to compile LLM pipeline: {e}")
            raise HTTPException(status_code=500, detail="Failed to compile LLM pipeline.")

    # ===== END NEW METHOD =====

# ===== Updated generate_syllabus() to use the new hybrid pipeline =====
def generate_syllabus(request_args: SyllabusRequestArgs, verbose=True):
    try:
        pipeline = SyllabusGeneratorPipeline(verbose=verbose)
        # Compile the new hybrid pipeline (sequential + parallel)
        pipeline.compile_sequential()

        # Convert request args to dictionary
        request_dict = request_args.to_dict()

        # --- Step 1: Generate course_information ---
        course_information = pipeline.chain_course_information.invoke(request_dict)
        # Inject the output into the request for chained prompts
        request_dict["course_information"] = course_information

        # --- Step 2: Generate course_description_objectives ---
        course_description_objectives = pipeline.chain_course_description_objectives.invoke(request_dict)

        # --- Step 3: Generate course_content using the chained course_information ---
        course_content = pipeline.chain_course_content.invoke(request_dict)

        # --- Step 4: Generate policies_procedures ---
        policies_procedures = pipeline.chain_policies_procedures.invoke(request_dict)

        # --- Step 5: Generate parallel outputs for assessment, learning resources, course schedule ---
        parallel_outputs = pipeline.parallel_pipeline.invoke(request_dict)

        model = SyllabusSchema(
            course_information=course_information,
            course_description_objectives=course_description_objectives,
            course_content=course_content,
            policies_procedures=policies_procedures,
            assessment_grading_criteria=parallel_outputs["branches"]["assessment_grading_criteria"],
            learning_resources=parallel_outputs["branches"]["learning_resources"],
            course_schedule=parallel_outputs["branches"]["course_schedule"],
        )
        return dict(model)

    except Exception as e:
        logger.error(f"Failed to generate syllabus: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate syllabus from LLM.")


# ------------------ Existing Schema Definitions (unchanged) ------------------
class CourseInformation(BaseModel):
    course_title: str = Field(description="The course title")
    grade_level: str = Field(description="The grade level")
    description: str = Field(description="The course description")

class CourseDescriptionObjectives(BaseModel):
    objectives: List[str] = Field(description="The course objectives")
    intended_learning_outcomes: List[str] = Field(description="The intended learning outcomes of the course")

class CourseContentItem(BaseModel):
    unit_time: str = Field(description="The unit of time for the course content")
    unit_time_value: int = Field(description="The unit of time value for the course content")
    topic: str = Field(description="The topic per unit of time for the course content")

class PoliciesProcedures(BaseModel):
    attendance_policy: str = Field(description="The attendance policy of the class")
    late_submission_policy: str = Field(description="The late submission policy of the class")
    academic_honesty: str = Field(description="The academic honesty policy of the class")

class AssessmentMethod(BaseModel):
    type_assessment: str = Field(description="The type of assessment")
    weight: int = Field(description="The weight of the assessment in the final grade")

class AssessmentGradingCriteria(BaseModel):
    assessment_methods: List[AssessmentMethod] = Field(description="The assessment methods")
    grading_scale: dict = Field(description="The grading scale")

class LearningResource(BaseModel):
    title: str = Field(description="The book title of the learning resource")
    author: str = Field(description="The book author of the learning resource")
    year: int = Field(description="The year of creation of the book")

class CourseScheduleItem(BaseModel):
    unit_time: str = Field(description="The unit of time for the course schedule item")
    unit_time_value: int = Field(description="The unit of time value for the course schedule item")
    date: str = Field(description="The date for the course schedule item")
    topic: str = Field(description="The topic for the learning resource")
    activity_desc: str = Field(description="The descrition of the activity for the learning resource")

class SyllabusSchema(BaseModel):
    course_information: CourseInformation = Field(description="The course information")
    course_description_objectives: CourseDescriptionObjectives = Field(description="The objectives of the course")
    course_content: List[CourseContentItem] = Field(description="The content of the course")
    policies_procedures: PoliciesProcedures = Field(description="The policies procedures of the course")
    assessment_grading_criteria: AssessmentGradingCriteria = Field(description="The asssessment grading criteria of the course")
    learning_resources: List[LearningResource] = Field(description="The learning resources of the course")
    course_schedule: List[CourseScheduleItem] = Field(description="The course schedule")
