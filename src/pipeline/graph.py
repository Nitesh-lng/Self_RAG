from langgraph.graph import StateGraph, START, END
from src.pipeline.state import GraphState
from src.pipeline.nodes import (
    retrieve, grade_documents, generate, transform_query,
    decide_to_generate, grade_generation,)

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not_supported": "generate",     
        "not_useful": "transform_query", 
        "useful": END,                   
    },
)

app = workflow.compile()