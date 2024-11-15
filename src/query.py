from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete


# Configura la directory di lavoro per l'istanza LightRAG
WORKING_DIR = "src/output/cost_out_v10(atomized_it+man)"

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    #llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

""" print("="*500)
print(rag.query("enti locali", param=QueryParam(mode="naive")))
print("="*500)
# Perform local search
print(rag.query("enti locali", param=QueryParam(mode="local")))
print("="*500)
# Perform global search
print(rag.query("enti locali", param=QueryParam(mode="global"))) """

print("="*500)
# Perform hybrid search
print(rag.query("enti locali", param=QueryParam(mode="hybrid")))
print("="*500)