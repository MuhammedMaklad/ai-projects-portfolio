# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: mlflow-dev-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## LLM RAG Evaluation with MLflow Example Notebook

# %% [raw]
# <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/rag/notebooks/mlflow-e2e-evaluation.ipynb" class="notebook-download-btn"><i class="fas fa-download"></i>Download this Notebook</a>

# %% [markdown]
# Welcome to this comprehensive tutorial on evaluating Retrieval-Augmented Generation (RAG) systems using MLflow. This tutorial is designed to guide you through the intricacies of assessing various RAG systems, focusing on how they can be effectively integrated and evaluated in a real-world context. Whether you are a data scientist, a machine learning engineer, or simply an enthusiast in the field of AI, this tutorial offers valuable insights and practical knowledge.
#
# ### What You Will Learn:
#
# 1. **Setting Up the Environment**:
#    - Learn how to set up your development environment with all the necessary tools and libraries, including MLflow, OpenAI, ChromaDB, LangChain, and more. This section ensures you have everything you need to start working with RAG systems.
#
# 2. **Understanding RAG Systems**:
#    - Delve into the concept of Retrieval-Augmented Generation and its significance in modern AI applications. Understand how RAG systems leverage both retrieval and generation capabilities to provide accurate and contextually relevant responses.
#
# 3. **Securely Managing API Keys with Databricks Secrets**:
#    - Explore the best practices for securely managing API keys using Databricks Secrets. This part is crucial for ensuring the security and integrity of your application.
#
# 4. **Deploying and Testing RAG Systems with MLflow**:
#    - Learn how to create, deploy, and test RAG systems using MLflow. This includes setting up endpoints, deploying models, and querying them to see their responses in action.
#
# 5. **Evaluating Performance with MLflow**: 
#    - Dive into evaluating the RAG systems using MLflow's evaluation tools. Understand how to use metrics like relevance and latency to assess the performance of your RAG system.
#
# 6. **Experimenting with Chunking Strategies**:
#    - Experiment with different text chunking strategies to optimize the performance of RAG systems. Understand how the size of text chunks affects retrieval accuracy and system responsiveness.
#
# 7. **Creating and Using Evaluation Datasets**:
#    - Learn how to create and utilize evaluation datasets (Golden Datasets) to effectively assess the performance of your RAG system.
#
# 8. **Combining Retrieval and Generation for Question Answering**:
#    - Gain insights into how retrieval and generation components work together in a RAG system to answer questions based on a given context or documentation.
#
#
# By the end of this tutorial, you will have a thorough understanding of how to evaluate and optimize RAG systems using MLflow. You will be equipped with the knowledge to deploy, test, and refine RAG systems, making them suitable for various practical applications. This tutorial is your stepping stone into the world of advanced AI model evaluation and deployment.

# %% jupyter={"outputs_hidden": true}
# %pip install mlflow>=2.8.1
# %pip install openai
# %pip install chromadb==0.4.15
# %pip install langchain==0.0.348
# %pip install tiktoken
# %pip install 'mlflow[genai]'
# %pip install databricks-sdk --upgrade

# %%
dbutils.library.restartPython()  # noqa: F821

# %%
import ast
import os
from typing import List

import chromadb
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.databricks import DatabricksEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import Databricks
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import mlflow
import mlflow.deployments
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance

# %%
# check mlflow version
mlflow.__version__

# %%
# check chroma version
chromadb.__version__

# %% [markdown]
# ### Set-up Databricks Workspace Secrets
#
# In order to use the secrets that are defined within this notebook, ensure that they are set via following the [guide to Databricks Secrets here](https://docs.databricks.com/en/security/secrets/secrets.html). It is highly recommended to utilize the [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html) to set secrets within your workspace for a secure experience.
#
# In order to safely store and access your API KEY for Azure OpenAI, ensure that you are setting the following when registering your secret:
#
# - **KEY_NAME**: The name that you will be setting for your Azure OpenAI Key
# - **SCOPE_NAME**: The referenced scope that your secret will reside in, within Databricks Secrets
# - **OPENAI_API_KEY**: Your Azure OpenAI Key
#
# As an example, you would set these keys through a terminal as follows:
#
# ```bash
#     databricks secrets put-secret "<SCOPE_NAME>" "<KEY_NAME>" --string-value "<OPENAI_API_KEY>"
# ```

# %%
# Set your Scope and Key Names that you used when registering your API KEY from the Databricks CLI
# Do not put your OpenAI API Key in the notebook!
SCOPE_NAME = ...
KEY_NAME = ...

# %%
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope=SCOPE_NAME, key=KEY_NAME)  # noqa: F821
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# Ensure that you set the name of your OPEN_API_BASE value to the name of your OpenAI instance on Azure
os.environ["OPENAI_API_BASE"] = "https://<NAME_OF_YOUR_INSTANCE>.openai.azure.com/"  # replace this!
os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-mini"
os.environ["OPENAI_ENGINE"] = "gpt-4o-mini"

# %% [markdown]
# ### Create and Test Endpoint on MLflow for OpenAI

# %%
client = mlflow.deployments.get_deploy_client("databricks")

endpoint_name = "<your-endpoint-name>"  # replace this!
client.create_endpoint(
    name=endpoint_name,
    config={
        "served_entities": [
            {
                "name": "test-gpt",  # Provide a unique identifying name for your deployments endpoint
                "external_model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "task": "llm/v1/completions",
                    "openai_config": {
                        "openai_api_type": "azure",
                        # replace with your own secrets, for reference see https://docs.databricks.com/en/security/secrets/secrets.html
                        "openai_api_key": "{{secrets/scope/openai_api_key}}",
                        "openai_api_base": "{{secrets/scope/openai_api_base}}",
                        "openai_deployment_name": "gpt-4o-mini",
                        "openai_api_version": "2023-05-15",
                    },
                },
            }
        ],
    },
)

# %%
print(
    client.predict(
        endpoint=endpoint_name,
        inputs={
            "prompt": "How is Pi calculated? Be very concise.",
            "max_tokens": 100,
        },
    )
)

# %% [markdown]
# ### Create RAG POC with LangChain and log with MLflow
#
# Use Langchain and Chroma to create a RAG system that answers questions based on the MLflow documentation.

# %%
loader = WebBaseLoader(
    [
        "https://mlflow.org/docs/latest/index.html",
        "https://mlflow.org/docs/latest/tracking/autolog.html",
        "https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
        "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html",
    ]
)

documents = loader.load()
CHUNK_SIZE = 1000
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

llm = Databricks(
    endpoint_name="<your-endpoint-name>",  # replace this!
    extra_params={
        "temperature": 0.1,
        "top_p": 0.1,
        "max_tokens": 500,
    },  # parameters used in AI Playground
)


# create the embedding function using Databricks Foundation Model APIs
embedding_function = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
docsearch = Chroma.from_documents(texts, embedding_function)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(fetch_k=3),
    return_source_documents=True,
)

# %% [markdown]
# ### Evaluate the Vector Database and Retrieval using `mlflow.evaluate()`
#
# #### Create an eval dataset (Golden Dataset)
#
# We can [leveraging the power of an LLM to generate synthetic data for testing](https://mlflow.org/docs/latest/llms/rag/notebooks/question-generation-retrieval-evaluation.html), offering a creative and efficient alternative. To our readers and customers, we emphasize the importance of crafting a dataset that mirrors the expected inputs and outputs of your RAG application. It's a journey worth taking for the incredible insights you'll gain!

# %% jupyter={"outputs_hidden": true}
EVALUATION_DATASET_PATH = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/llms/RAG/static_evaluation_dataset.csv"

synthetic_eval_data = pd.read_csv(EVALUATION_DATASET_PATH)

# Load the static evaluation dataset from disk and deserialize the source and retrieved doc ids
synthetic_eval_data["source"] = synthetic_eval_data["source"].apply(ast.literal_eval)
synthetic_eval_data["retrieved_doc_ids"] = synthetic_eval_data["retrieved_doc_ids"].apply(
    ast.literal_eval
)

# %%
display(synthetic_eval_data)

# %% [markdown]
# ### Evaluating the Embedding Model with MLflow
#
# In this part of the tutorial, we focus on evaluating the embedding model's performance in the context of a retrieval system. The process involves a series of steps to assess how effectively the model can retrieve relevant documents based on given questions.
#
# #### Creating Evaluation Data
# - We start by defining a set of questions and their corresponding source URLs. This `eval_data` DataFrame acts as our evaluation dataset, allowing us to test the model's ability to link questions to the correct source documents.
#
# #### The `evaluate_embedding` Function
# - The `evaluate_embedding` function is designed to assess the performance of a given embedding function.
# - **Chunking Strategy**: The function begins by splitting a list of documents into chunks using a `CharacterTextSplitter`. The size of these chunks is crucial, as it can influence the retrieval accuracy.
# - **Retriever Initialization**: We then use `Chroma.from_documents` to create a retriever with the specified embedding function. This retriever is responsible for finding documents relevant to a given query.
# - **Retrieval Process**: The function defines a `retriever_model_function` that applies the retriever to each question in the evaluation dataset. It retrieves document IDs that the model finds most relevant for each question.
#
# #### MLflow Evaluation
# - With `mlflow.start_run()`, we initiate an evaluation run. `mlflow.evaluate` is then called to evaluate our retriever model function against the evaluation dataset.
# - We use the default evaluator with specified targets to assess the model's performance.
# - The results of this evaluation, stored in `eval_results_of_retriever_df_bge`, are displayed, providing insights into the effectiveness of the embedding model in document retrieval.
#
# #### Further Evaluation with Metrics
# - Additionally, we perform a more detailed evaluation using various metrics like precision, recall, and NDCG at different 'k' values. These metrics offer a deeper understanding of the model's retrieval accuracy and ranking effectiveness.
#
# This evaluation step is integral to understanding the strengths and weaknesses of our embedding model in a real-world RAG system. By analyzing these results, we can make informed decisions about model adjustments or optimizations to improve overall system performance.
#

# %%
eval_data = pd.DataFrame(
    {
        "question": [
            "What is MLflow?",
            "What is Databricks?",
            "How to serve a model on Databricks?",
            "How to enable MLflow Autologging for my workspace by default?",
        ],
        "source": [
            ["https://mlflow.org/docs/latest/index.html"],
            ["https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html"],
            ["https://mlflow.org/docs/latest/python_api/mlflow.deployments.html"],
            ["https://mlflow.org/docs/latest/tracking/autolog.html"],
        ],
    }
)


# %%
def evaluate_embedding(embedding_function):
    CHUNK_SIZE = 1000
    list_of_documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    docs = text_splitter.split_documents(list_of_documents)
    retriever = Chroma.from_documents(docs, embedding_function).as_retriever()

    def retrieve_doc_ids(question: str) -> List[str]:
        docs = retriever.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]

    def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
        return question_df["question"].apply(retrieve_doc_ids)

    with mlflow.start_run():
        return mlflow.evaluate(
            model=retriever_model_function,
            data=eval_data,
            model_type="retriever",
            targets="source",
            evaluators="default",
        )


result1 = evaluate_embedding(DatabricksEmbeddings(endpoint="databricks-bge-large-en"))
# To validate the results of a different model, comment out the above line and uncomment the below line:
# result2 = evaluate_embedding(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

eval_results_of_retriever_df_bge = result1.tables["eval_results_table"]
# To validate the results of a different model, comment out the above line and uncomment the below line:
# eval_results_of_retriever_df_MiniLM = result2.tables["eval_results_table"]
display(eval_results_of_retriever_df_bge)

# %% [markdown]
# ### Evaluate different Top K strategy with MLflow

# %%
with mlflow.start_run() as run:
    evaluate_results = mlflow.evaluate(
        data=eval_results_of_retriever_df_bge,
        targets="source",
        predictions="outputs",
        evaluators="default",
        extra_metrics=[
            mlflow.metrics.precision_at_k(1),
            mlflow.metrics.precision_at_k(2),
            mlflow.metrics.precision_at_k(3),
            mlflow.metrics.recall_at_k(1),
            mlflow.metrics.recall_at_k(2),
            mlflow.metrics.recall_at_k(3),
            mlflow.metrics.ndcg_at_k(1),
            mlflow.metrics.ndcg_at_k(2),
            mlflow.metrics.ndcg_at_k(3),
        ],
    )

display(evaluate_results.tables["eval_results_table"])


# %% [markdown]
# ### Evaluate the Chunking Strategy with MLflow
#
# In the realm of RAG systems, the strategy for dividing text into chunks plays a pivotal role in both retrieval effectiveness and the overall system performance. Let's delve into why and how we evaluate different chunking strategies:
#
# #### Importance of Chunking:
# - **Influences Retrieval Accuracy**: The way text is chunked can significantly affect the retrieval component of RAG systems. Smaller chunks may lead to more focused and relevant document retrieval, while larger chunks might capture broader context.
# - **Impacts System's Responsiveness**: The size of text chunks also influences the speed of document retrieval and processing. Smaller chunks can be processed more quickly but may require the system to evaluate more chunks overall.
#
# #### Evaluating Different Chunk Sizes:
# - **Purpose**: By evaluating different chunk sizes, we aim to find an optimal balance between retrieval accuracy and processing efficiency. This involves experimenting with various chunk sizes to see how they impact the system's performance.
# - **Method**: We create text chunks of different sizes (e.g., 1000 characters, 2000 characters) and then evaluate how each chunking strategy affects the RAG system. Key aspects to observe include the relevance of retrieved documents and the system's latency.
#
# In this example below, we're using the default evaluation suite to provide a comprehensive adjudication of the quality of the responses to retrieved document contents to determine what the impact to the quality of the returned references are, allowing us to explore and tune the chunk size in order to arrive at a configuration that best handles our suite of test questions.
#
# Note that the embedding model has changed in this next code block. Above, we were using ``DatabricksEmbeddings(endpoint="databricks-bge-large-en")``, while now we're evaluating the performance of ``SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")``

# %%
def evaluate_chunk_size(chunk_size):
    list_of_documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    docs = text_splitter.split_documents(list_of_documents)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = Chroma.from_documents(docs, embedding_function).as_retriever()

    def retrieve_doc_ids(question: str) -> List[str]:
        docs = retriever.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]

    def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
        return question_df["question"].apply(retrieve_doc_ids)

    with mlflow.start_run():
        return mlflow.evaluate(
            model=retriever_model_function,
            data=eval_data,
            model_type="retriever",
            targets="source",
            evaluators="default",
        )


result1 = evaluate_chunk_size(1000)
result2 = evaluate_chunk_size(2000)


display(result1.tables["eval_results_table"])
display(result2.tables["eval_results_table"])


# %% [markdown]
# ### Evaluate the RAG system using `mlflow.evaluate()`
#
# In this section, we'll delve into evaluating the Retrieval-Augmented Generation (RAG) systems using `mlflow.evaluate()`. This evaluation is crucial for assessing the effectiveness and efficiency of RAG systems in question-answering contexts. We focus on two key metrics: `relevance_metric` and `latency`.
#
# #### Relevance Metric:
# - **What It Measures**: The `relevance_metric` quantifies how relevant the RAG system's answers are to the input questions. This metric is critical for understanding the accuracy and contextual appropriateness of the system's responses.
# - **Why It's Important**: In question-answering systems, relevance is paramount. The ability of a RAG system to provide accurate and contextually correct answers determines its utility and effectiveness in real-world applications, such as information retrieval and customer support.
# - **Tutorial Context**: Within our tutorial, we utilize the `relevance_metric` to evaluate the quality of answers provided by the RAG system. It serves as a quantitative measure of the system's content accuracy, reflecting its capability to generate useful and precise responses.
#
# #### Latency:
# - **What It Measures**: The `latency` metric captures the response time of the RAG system. It measures the duration taken by the system to generate an answer after receiving a query.
# - **Why It's Important**: Response time is a critical factor in user experience. In interactive systems, lower latency leads to a more efficient and satisfying user experience. High latency, conversely, can be detrimental to user satisfaction.
# - **Tutorial Context**: In this tutorial, we assess the system's efficiency in terms of response time through the `latency` metric. This evaluation is vital for understanding the system's performance in a production environment, where timely responses are as important as their accuracy.
#
# To start with evaluating, we'll create a simple function that runs each input through the RAG chain

# %%
def model(input_df):
    return input_df["questions"].map(qa).tolist()


# %% [markdown]
# ### Create an evaluation dataset (Golden Dataset)

# %%
eval_df = pd.DataFrame(
    {
        "questions": [
            "What is MLflow?",
            "What is Databricks?",
            "How to serve a model on Databricks?",
            "How to enable MLflow Autologging for my workspace by default?",
        ],
    }
)
display(eval_df)

# %% [markdown]
# ### Evaluate using LLM as a Judge and Basic Metrics
#
# In this concluding section of the tutorial, we perform a final evaluation of our RAG system using MLflow's powerful evaluation tools. This evaluation is crucial for assessing the performance and efficiency of the question-answering model.
#
# #### Key Steps in the Evaluation:
#
# 1. **Setting the Deployment Target**:
#    - The deployment target is set to Databricks, enabling us to retrieve all endpoints in the Databricks Workspace. This is essential for accessing our deployed models.
#
# 2. **Relevance Metric Setup**:
#    - We initialize the `relevance` metric using a model hosted on Databricks. This metric assesses how relevant the answers generated by our RAG system are in response to the input questions.
#
# 3. **Running the Evaluation**:
#    - An MLflow run is initiated, and `mlflow.evaluate()` is called to evaluate our RAG model against the prepared evaluation dataset.
#    - The model is evaluated as a "question-answering" system using default evaluators.
#    - Additional metrics, including the `relevance_metric` and `latency`, are specified. These metrics provide insights into the relevance of the answers and the response time of the model.
#    - The `evaluator_config` maps the input questions and context, ensuring the correct evaluation of the RAG system.
#
# 4. **Results and Metrics Display**:
#    - The results of the evaluation, including key metrics, are displayed in a table format, providing a clear and structured view of the model's performance based on relevance and latency.
#
# This comprehensive evaluation step is vital for understanding the effectiveness and efficiency of our RAG system. By assessing both the relevance of the answers and the latency of the responses, we gain a holistic view of the model's performance, guiding any further optimization or deployment decisions.

# %%
set_deployments_target("databricks")  # To retrieve all endpoint in your Databricks Workspace

relevance_metric = relevance(
    model="endpoints:/databricks-llama-2-70b-chat"
)  # You can also use any model you have hosted on Databricks, models from the Marketplace or models in the Foundation model API

with mlflow.start_run():
    results = mlflow.evaluate(
        model,
        eval_df,
        model_type="question-answering",
        evaluators="default",
        predictions="result",
        extra_metrics=[relevance_metric, mlflow.metrics.latency()],
        evaluator_config={
            "col_mapping": {
                "inputs": "questions",
                "context": "source_documents",
            }
        },
    )
    print(results.metrics)

display(results.tables["eval_results_table"])
