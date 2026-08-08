from src.config.settings import LLM_MODEL, INDEX_PATH, TOP_K, MAX_RETRIES
from src.entity.schemas import GradeDocuments, GradeHallucination, GradeAnswer
from src.components.vector_store import VectorStoreBuilder
from src.utils.logger import get_logger
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
load_dotenv()
logger = get_logger(__name__)

llm = ChatGroq(model=LLM_MODEL, temperature=0)

_vector_store=None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreBuilder().load(INDEX_PATH)
    return _vector_store

#ISREL, ISSUP, ISUSE

structured_doc_grader=llm.with_structured_output(GradeDocuments)
doc_grade_prompt=ChatPromptTemplate.from_template(
    """You are a grader assessing whether a retrieved document is relevant to a user question.
    The document is relevant ONLY if it can actually help answer the specific question asked.
    Do not grade as relevant just because it shares a keyword.
    Give a binary score 'yes' or 'no'.

    Document: {document}
    Question: {question}"""
)
doc_grader = doc_grade_prompt | structured_doc_grader

structured_hallucination_grader=llm.with_structured_output(GradeHallucination)
hallucination_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an LLM generation is grounded in / supported by the retrieved documents.
    Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in the documents.

    Documents: {documents}
    Generation: {generation}"""
)
hallucination_grader = hallucination_prompt | structured_hallucination_grader

structured_answer_grader = llm.with_structured_output(GradeAnswer)
answer_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an answer addresses / resolves the user question.
    Give a binary score 'yes' or 'no'. 'yes' means the answer addresses the question.

    Question: {question}
    Generation: {generation}"""
)
answer_grader = answer_prompt | structured_answer_grader

gen_prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
    If the context does not contain the answer, say "I don't have enough information to answer that."

    Context: {context}
    Question: {question}
    Answer:"""
)
rag_chain = gen_prompt | llm | StrOutputParser()

def retrieve(state):
    logger.info("---RETRIEVE---")
    question = state['question']
    documents = get_vector_store().similarity_search(question,k=TOP_K)
    return {'documents' : documents}

def grade_documents(state):
    logger.info("---GRADE DOCUMENTS---")
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    for doc in documents:
        result = doc_grader.invoke({'document': doc.page_content , 'question': question})
        if result.binary_score.lower() == 'yes':
            filtered_docs.append(doc)

    return {'documents':filtered_docs}

def generate(state):
    logger.info("---GENERATE---")
    documents = state['documents']
    question = state['question']
    retries = state.get('retries',0)

    context = '\n\n'.join(doc.page_content for doc in documents)
    generation = rag_chain.invoke({'context': context, 'question': question})

    return {'generation': generation, 'retries': retries+1}

def decide_to_generate(state):
    logger.info("---DECIDE TO GENERATE---")
    documents = state['documents']
    retries = state.get('retries',0)

    if not documents:
        if retries >= MAX_RETRIES:
            logger.info("---NO DOCS, MAX RETRIES: giving up---")
            return generate
        
        logger.info("---NO RELEVANT DOCS: re-retrieve (transform query)---")
        return 'transform query'
    else:
        return generate
    
def grade_generation(state):
    logger.info("---GRADE GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retries = state.get("retries", 0)

    if retries >= MAX_RETRIES:
        logger.info("---MAX RETRIES REACHED: accepting answer---")
        return "useful"

    docs_text = "\n\n".join(doc.page_content for doc in documents)

    # Check 1: hallucination (grounded?)
    hall = hallucination_grader.invoke({"documents": docs_text, "generation": generation})
    if hall.binary_score.lower() == "no":
        logger.info("---HALLUCINATION: not grounded, regenerate---")
        return "not_supported"

    # Check 2: answer (addresses question?)
    ans = answer_grader.invoke({"question": question, "generation": generation})
    if ans.binary_score.lower() == "yes":
        logger.info("---USEFUL: answer addresses question---")
        return "useful"
    else:
        logger.info("---NOT USEFUL: re-retrieve---")
        return "not_useful"
    
transform_prompt = ChatPromptTemplate.from_template(
    """You are rewriting a user question to improve document retrieval.
    Look at the question and rewrite it to be clearer and more search-optimized.
    Return ONLY the rewritten question, nothing else.

    Question: {question}"""
)
transform_chain = transform_prompt | llm | StrOutputParser()

def transform_query(state):
    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    better_question = transform_chain.invoke({"question": question})
    
    return {"question": better_question}