"""
FastAPI Server for Answer Evaluator Pipeline.

Receives runtime inputs (question, reference_answer, student_answer) via API endpoint
and processes semantic similarity, concept coverage, scoring, and LLM report generation.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from embeddings import get_model
from evaluator import evaluate
from openrouter import generate_report, OpenRouterError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the embedding model on server startup so API evaluation calls execute quickly.
    """
    print("[Server Startup] Pre-loading sentence-transformer embedding model...")
    get_model()
    print("[Server Startup] Model loaded successfully. Ready to accept evaluation requests.")
    yield
    print("[Server Shutdown] Shutting down Answer Evaluator server.")


app = FastAPI(
    title="Answer Evaluator API",
    description="API server for evaluating student descriptive answers against reference answers.",
    version="1.0.0",
    lifespan=lifespan,
)


class EvaluationRequest(BaseModel):
    question: str = Field(..., description="The question text", example="Discuss in detail about the types of Aquifers")
    reference_answer: Optional[str] = Field(None, description="The reference model answer")
    reference: Optional[str] = Field(None, description="Alias for reference model answer")
    student_answer: str = Field(..., description="The student answer to be evaluated")

    def get_reference(self) -> str:
        return (self.reference_answer or self.reference or "").strip()



class ConceptDetail(BaseModel):
    concept: str
    similarity: float
    status: str


class EvaluationResponse(BaseModel):
    score: float
    similarity: float
    concept_coverage: float
    concepts: List[ConceptDetail]
    report: Optional[Dict[str, Any]] = None
    llm_error: Optional[str] = None


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Answer Evaluator API",
        "version": "1.0.0"
    }


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
def evaluate_answer(req: EvaluationRequest):
    """
    Evaluates a student's answer against a reference answer and question provided in runtime payload.
    """
    question = req.question.strip()
    reference_answer = req.get_reference()
    student_answer = req.student_answer.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty."
        )
    if not reference_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reference answer field cannot be empty (provide reference_answer or reference)."
        )
    if not student_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student answer field cannot be empty."
        )

    try:
        result = evaluate(question, reference_answer, student_answer)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        )

    response_data = {
        "score": result["score"],
        "similarity": result["similarity"],
        "concept_coverage": result["concept_coverage"],
        "concepts": result["concepts"],
        "report": None,
        "llm_error": None
    }

    try:
        report = generate_report(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
            concepts_result=result["concepts"],
            final_score=result["score"],
        )
        response_data["report"] = report
    except OpenRouterError as exc:
        response_data["llm_error"] = str(exc)

    return response_data
