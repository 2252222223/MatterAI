from CEO.Base.CEO_sk import sk
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def expert_experience_match(goals, last_feedback, Expert_experience_path):
    embeddings = OpenAIEmbeddings(openai_api_key=sk)
    # 加载数据
    expert_vector = Chroma(
        persist_directory=Expert_experience_path,
        embedding_function=embeddings)
    Experience_Match = goals[0] + last_feedback
    Expert_experience_match = expert_vector.similarity_search(Experience_Match, k=1, include_metadata=False)

    expert_experience = Expert_experience_match[0].page_content

    return expert_experience
