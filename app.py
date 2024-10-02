import re
import json
import nltk
import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from docarray import DocList
from docarray.index import InMemoryExactNNIndex
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from common.repo_doc import RepoDoc
from nltk.stem import WordNetLemmatizer

from similarityCal.utils import calculate_similarity

nltk.download("wordnet")
KMEANS_TOPIC_MODEL_PATH = Path(__file__).parent.joinpath("data/kmeans_model_topic_scibert.pkl")
KMEANS_CODE_MODEL_PATH = Path(__file__).parent.joinpath("data/kmeans_model_code_unixcoder.pkl")

SCIBERT_MODEL_PATH = "allenai/scibert_scivocab_uncased"
# SCIBERT_MODEL_PATH = Path(__file__).parent.joinpath("data/scibert_scivocab_uncased")  # Download locally

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 1. Product environment
INDEX_PATH = Path(__file__).parent.joinpath("data/index.bin")
TOPIC_CLUSTER_PATH = Path(__file__).parent.joinpath("data/repo_topic_clusters.json")
CODE_CLUSTER_PATH = Path(__file__).parent.joinpath("data/repo_code_clusters.json")

# 2. Developing environment
# INDEX_PATH = Path(__file__).parent.joinpath("data/index_developing.bin")
# TOPIC_CLUSTER_PATH = Path(__file__).parent.joinpath("data/repo_topic_clusters_developing.json")
# CODE_CLUSTER_PATH = Path(__file__).parent.joinpath("data/repo_code_clusters_developing.json")


@st.cache_resource(show_spinner="Loading repositories basic information...")
def load_index():
    """
    The function to load the index file and return a RepoDoc object with default value
    :return: index and a RepoDoc object with default value
    """
    default_doc = RepoDoc(
        name="",
        topics=[],
        stars=0,
        license="",
        code_embedding=None,
        doc_embedding=None,
        readme_embedding=None,
        requirement_embedding=None,
        repository_embedding=None
    )

    return InMemoryExactNNIndex[RepoDoc](index_file_path=INDEX_PATH), default_doc


@st.cache_resource(show_spinner="Loading repositories topic clusters...")
def load_repo_topic_clusters():
    """
    The function to load the repo-topic_clusters file
    :return: a dictionary with the repo-topic_clusters
    """
    with open(TOPIC_CLUSTER_PATH, "r") as file:
        repo_topic_clusters = json.load(file)

    return repo_topic_clusters


@st.cache_resource(show_spinner="Loading repositories code clusters...")
def load_repo_code_clusters():
    """
    The function to load the repo-code_clusters file
    :return: a dictionary with the repo-code_clusters
    """
    with open(CODE_CLUSTER_PATH, "r") as file:
        repo_code_clusters = json.load(file)

    return repo_code_clusters


@st.cache_resource(show_spinner="Loading RepoSim4Py pipeline model...")
def load_pipeline_model():
    """
    The function to load RepoSim4Py pipeline model
    :return: a HuggingFace pipeline
    """
    # Option 1 --- Download model by HuggingFace username/model_name
    model_path = "Henry65/RepoSim4Py"

    # Option 2 --- Download model locally
    # model_path = Path(__file__).parent.joinpath("data/RepoSim4Py")

    return pipeline(
        model=model_path,
        trust_remote_code=True,
        device_map="auto"
    )


@st.cache_resource(show_spinner="Loading SciBERT model...")
def load_scibert_model():
    """
    The function to load SciBERT model
    :return: tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_PATH)
    scibert_model = AutoModel.from_pretrained(SCIBERT_MODEL_PATH).to(device)

    return tokenizer, scibert_model


@st.cache_resource(show_spinner="Loading KMeans model (topic clusters)...")
def load_topic_kmeans_model():
    """
    The function to load KMeans model (topic clusters)
    :return: a KMeans model (topic clusters)
    """
    return joblib.load(KMEANS_TOPIC_MODEL_PATH)


@st.cache_resource(show_spinner="Loading KMeans model (code clusters)...")
def load_code_kmeans_model():
    """
    The function to load KMeans model (code clusters)
    :return: a KMeans model (code clusters)
    """
    return joblib.load(KMEANS_CODE_MODEL_PATH)


@st.cache_resource(show_spinner="Loading SimilarityCal model...")
def load_similaritycal_model(mode: str):
    """
    The function to load SimilarityCal model
    mode: 'code' or 'topic'
    :return: the SimilarityCal model
    """
    if mode == 'topic':
        sim_cal_model = torch.load('similarityCal/topic.pt')
    elif mode == 'code':
        sim_cal_model = torch.load('similarityCal/code.pt')
    else:
        raise ValueError("parameter 'mode' must be 'code' or 'topic'")
    sim_cal_model.to(device)
    sim_cal_model.eval()
    return sim_cal_model


def generate_scibert_embedding(tokenizer, scibert_model, text):
    """
    The function for generating SciBERT embeddings based on topic text
    :param text: the topic text
    :return: topic embeddings
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = scibert_model(**inputs)
    # Use mean pooling for sentence representation
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()

    return embeddings


@st.cache_data(show_spinner=False)
def run_pipeline_model(_model, repo_name, github_token):
    """
    The function to generate repo_info by using pipeline model
    :param _model: pipeline
    :param repo_name: the name of repository
    :param github_token: GitHub token
    :return: the information generated by the pipeline
    """
    with st.spinner(
            f"Downloading and extracting the {repo_name}, this may take a while..."
    ):
        extracted_infos = _model.preprocess(repo_name, github_token=github_token)

    if not extracted_infos:
        return None

    st_proress_bar = st.progress(0.0)
    with st.spinner(f"Generating embeddings for {repo_name}..."):
        repo_info = _model.forward(extracted_infos, st_progress=st_proress_bar)[0]
    st_proress_bar.empty()

    return repo_info


def run_index_search(index, query, search_field, limit):
    """
    The function to search at index file based on query and limit
    :param index: the index
    :param query: query
    :param search_field: which field to search for
    :param limit: page limit
    :return: a dataframe with search results
    """
    top_matches, scores = index.find(
        query=query, search_field=search_field, limit=limit
    )

    search_results = top_matches.to_dataframe()
    search_results["scores"] = scores

    return search_results


def run_topic_cluster_search(repo_topic_clusters, repo_name_list):
    """
    The function to search topic cluster number for such repositories.
    :param repo_topic_clusters: dictionary with repo-topic_clusters
    :param repo_name_list: list or array represent repository names
    :return: topic cluster number list
    """
    topic_clusters = []
    for repo_name in repo_name_list:
        topic_clusters.append(repo_topic_clusters[repo_name])

    return topic_clusters


def run_code_cluster_search(repo_code_clusters, repo_name_list):
    """
    The function to search code cluster number for such repositories.
    :param repo_code_clusters: dictionary with repo-code_clusters
    :param repo_name_list: list or array represent repository names
    :return: code cluster number list
    """
    code_clusters = []
    for repo_name in repo_name_list:
        code_clusters.append(repo_code_clusters[repo_name])

    return code_clusters


def run_similaritycal_search(index, repo_clusters, model, query_doc, query_cluster_number, limit):
    """
    The function to run SimilarityCal model.
    :param index: index file
    :param repo_clusters: repo-clusters (topic_cluster or code_cluster) json file
    :param model: SimilarityCal model
    :param query_doc: query repo doc
    :param query_cluster_number: query repo cluster number (code or topic)
    :param limit: limit
    :return: result dataframe
    """
    docs = index._docs
    result_dl = DocList[RepoDoc]()
    e1_list, e2_list = [], []
    for doc in docs:
        if query_cluster_number != repo_clusters[doc.name]:
            continue
        if doc.name != query_doc.name:
            e1, e2 = (torch.Tensor(query_doc.repository_embedding),
                      torch.Tensor(doc.repository_embedding))
            e1_list.append(e1)
            e2_list.append(e2)
            result_dl.append(doc)

    e1_list = torch.stack(e1_list).to(device)
    e2_list = torch.stack(e2_list).to(device)
    model.eval()
    similarity_scores = calculate_similarity(model, e1_list, e2_list)[:, 1].cpu().detach().numpy()
    df = result_dl.to_dataframe()
    df["scores"] = similarity_scores

    sorted_df = df.sort_values(by='scores', ascending=False).reset_index(drop=True).head(limit)
    sorted_df["rankings"] = sorted_df["scores"].rank(ascending=False, method='first').astype(int)
    sorted_df.drop(columns="scores", inplace=True)

    return sorted_df


if __name__ == "__main__":
    # Loading dataset and models
    index, default_doc = load_index()
    repo_topic_clusters = load_repo_topic_clusters()
    repo_code_clusters = load_repo_code_clusters()
    pipeline_model = load_pipeline_model()
    lemmatizer = WordNetLemmatizer()
    tokenizer, scibert_model = load_scibert_model()
    topic_kmeans = load_topic_kmeans_model()
    code_kmeans = load_code_kmeans_model()

    # Setting the sidebar
    with st.sidebar:
        st.text_input(
            label="GitHub Token",
            key="github_token",
            type="password",
            placeholder="Paste your GitHub token here",
            help="Consider setting GitHub token to avoid hitting rate limits: https://docs.github.com/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token",
        )

        st.slider(
            label="Search results limit",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="search_results_limit",
            help="Limit the number of search results",
        )

        st.multiselect(
            label="Display columns",
            options=["scores", "name", "topics", "code cluster", "topic cluster", "stars", "license"],
            default=["scores", "name", "topics", "code cluster", "topic cluster", "stars", "license"],
            help="Select columns to display in the search results",
            key="display_columns",
        )

    # Setting the main content
    st.title("RepoSnipy")

    st.text_input(
        "Enter a GitHub repository URL or owner/repository (case-sensitive):",
        value="",
        max_chars=200,
        placeholder="numpy/numpy",
        key="repo_input",
    )

    st.checkbox(
        label="Add/Update this repository to the index",
        value=False,
        key="update_index",
        help="Encode the latest version of this repository and add/update it to the index",
    )

    # Setting the search button
    search = st.button("Search")
    # The regular expression for repository
    repo_regex = r"^((git@|http(s)?://)?(github\.com)(/|:))?(?P<owner>[\w.-]+)(/)(?P<repo>[\w.-]+?)(\.git)?(/)?$"

    if search:
        match_res = re.match(repo_regex, st.session_state.repo_input)
        # 1. Repository can be matched
        if match_res is not None:
            repo_name = f"{match_res.group('owner')}/{match_res.group('repo')}"
            records = index.filter({"name": {"$eq": repo_name}})
            # 1) Building the query information
            query_doc = default_doc.copy() if not records else records[0]
            # 2) Recording the topic and code cluster numbers
            topic_cluster_number = -1 if not records else repo_topic_clusters[repo_name]
            code_cluster_number = -1 if not records else repo_code_clusters[repo_name]

            # Importance 1 ---- situation need to update repository information and cluster numbers
            if st.session_state.update_index or not records:
                # 1) Updating repository information by using RepoSim4Py pipeline
                repo_info = run_pipeline_model(pipeline_model, repo_name, st.session_state.github_token)
                if repo_info is None:
                    st.error("Repository not found or invalid GitHub token!")
                    st.stop()

                query_doc.name = repo_info["name"]
                query_doc.topics = repo_info["topics"]
                query_doc.stars = repo_info["stars"]
                query_doc.license = repo_info["license"]
                query_doc.code_embedding = None if np.all(repo_info["mean_code_embedding"] == 0) else repo_info[
                    "mean_code_embedding"].reshape(-1)
                query_doc.doc_embedding = None if np.all(repo_info["mean_doc_embedding"] == 0) else repo_info[
                    "mean_doc_embedding"].reshape(-1)
                query_doc.readme_embedding = None if np.all(repo_info["mean_readme_embedding"] == 0) else repo_info[
                    "mean_readme_embedding"].reshape(-1)
                query_doc.requirement_embedding = None if np.all(repo_info["mean_requirement_embedding"] == 0) else \
                    repo_info["mean_requirement_embedding"].reshape(-1)
                query_doc.repository_embedding = None if np.all(repo_info["mean_repo_embedding"] == 0) else repo_info[
                    "mean_repo_embedding"].reshape(-1)

                # 2) Updating topic cluster number
                topics_text = ' '.join(
                    [lemmatizer.lemmatize(topic.lower().replace('-', ' ')) for topic in query_doc.topics if
                     topic.lower() not in ["python", "python3"]])
                topic_embeddings = generate_scibert_embedding(tokenizer, scibert_model, topics_text)
                topic_cluster_number = int(topic_kmeans.predict(topic_embeddings)[0])

                # 3) Updating code cluster number
                code_embeddings = np.zeros((768,),
                                           dtype=np.float32) if query_doc.code_embedding is None else query_doc.code_embedding
                code_cluster_number = int(code_kmeans.predict(code_embeddings.reshape(1, -1))[0])

            # Importance 2 ---- update index file and repository clusters (topic and code) files
            if st.session_state.update_index:
                if not query_doc.license:
                    st.warning(
                        "License is missing in this repository and will not be persisted!"
                    )
                elif (query_doc.code_embedding is None) and (query_doc.doc_embedding is None) and (
                        query_doc.requirement_embedding is None) and (query_doc.readme_embedding is None) and (
                        query_doc.repository_embedding is None):
                    st.warning(
                        "This repository has no such useful information (code, docstring, readme and requirement) extracted and will not be persisted!"
                    )
                else:
                    index.index(query_doc)
                    repo_topic_clusters[query_doc.name] = topic_cluster_number
                    repo_code_clusters[query_doc.name] = code_cluster_number

                    with st.spinner("Persisting the index and repository clusters (topic and code)..."):
                        index.persist(str(INDEX_PATH))
                        with open(TOPIC_CLUSTER_PATH, "w") as file:
                            json.dump(repo_topic_clusters, file, indent=4)
                        with open(CODE_CLUSTER_PATH, "w") as file:
                            json.dump(repo_code_clusters, file, indent=4)
                        st.success("Repository updated to the index!")

                    load_index.clear()
                    load_repo_topic_clusters.clear()
                    load_repo_code_clusters.clear()

            st.session_state["query_doc"] = query_doc
            st.session_state["topic_cluster_number"] = topic_cluster_number
            st.session_state["code_cluster_number"] = code_cluster_number

        # 2. Repository cannot be matched
        else:
            st.error("Invalid input!")

    # Starting to query
    if "query_doc" in st.session_state:
        query_doc = st.session_state.query_doc
        topic_cluster_number = st.session_state.topic_cluster_number
        code_cluster_number = st.session_state.code_cluster_number
        limit = st.session_state.search_results_limit

        # Showing the query repository information
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "name": query_doc.name,
                        "topics": query_doc.topics,
                        "code cluster": code_cluster_number,
                        "topic cluster": topic_cluster_number,
                        "stars": query_doc.stars,
                        "license": query_doc.license,
                    }
                ],
            )
        )

        display_columns = st.session_state.display_columns
        modified_display_columns = ["rankings" if col == "scores" else col for col in display_columns]
        code_sim_tab, doc_sim_tab, readme_sim_tab, requirement_sim_tab, repo_sim_tab, code_cluster_tab, topic_cluster_tab, = st.tabs(
            ["Code_sim", "Docstring_sim", "Readme_sim", "Requirement_sim",
             "Repository_sim", "Code_cluster_sim", "Topic_cluster_sim"])

        with code_sim_tab:
            if query_doc.code_embedding is not None:
                code_sim_res = run_index_search(index, query_doc, "code_embedding", limit)
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, code_sim_res["name"])
                code_sim_res["topic cluster"] = topic_cluster_numbers
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, code_sim_res["name"])
                code_sim_res["code cluster"] = code_cluster_numbers
                st.dataframe(code_sim_res[display_columns])
            else:
                st.error("No function code was extracted for this repository!")

        with doc_sim_tab:
            if query_doc.doc_embedding is not None:
                doc_sim_res = run_index_search(index, query_doc, "doc_embedding", limit)
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, doc_sim_res["name"])
                doc_sim_res["topic cluster"] = topic_cluster_numbers
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, doc_sim_res["name"])
                doc_sim_res["code cluster"] = code_cluster_numbers
                st.dataframe(doc_sim_res[display_columns])
            else:
                st.error("No function docstring was extracted for this repository!")

        with readme_sim_tab:
            if query_doc.readme_embedding is not None:
                readme_sim_res = run_index_search(index, query_doc, "readme_embedding", limit)
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, readme_sim_res["name"])
                readme_sim_res["topic cluster"] = topic_cluster_numbers
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, readme_sim_res["name"])
                readme_sim_res["code cluster"] = code_cluster_numbers
                st.dataframe(readme_sim_res[display_columns])
            else:
                st.error("No readme file was extracted for this repository!")

        with requirement_sim_tab:
            if query_doc.requirement_embedding is not None:
                requirement_sim_res = run_index_search(index, query_doc, "requirement_embedding", limit)
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, requirement_sim_res["name"])
                requirement_sim_res["topic cluster"] = topic_cluster_numbers
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, requirement_sim_res["name"])
                requirement_sim_res["code cluster"] = code_cluster_numbers
                st.dataframe(requirement_sim_res[display_columns])
            else:
                st.error("No requirement file was extracted for this repository!")

        with repo_sim_tab:
            if query_doc.repository_embedding is not None:
                # Repo Sim tab
                repo_sim_res = run_index_search(index, query_doc, "repository_embedding", limit)
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, repo_sim_res["name"])
                repo_sim_res["topic cluster"] = topic_cluster_numbers
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, repo_sim_res["name"])
                repo_sim_res["code cluster"] = code_cluster_numbers
                st.dataframe(repo_sim_res[display_columns])
            else:
                st.error("No such useful information was extracted for this repository!")

        with code_cluster_tab:
            if query_doc.repository_embedding is not None:
                sim_cal_model = load_similaritycal_model("code")
                cluster_df = run_similaritycal_search(index, repo_code_clusters, sim_cal_model,
                                                      query_doc, code_cluster_number, limit)
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, cluster_df["name"])
                cluster_df["code cluster"] = code_cluster_numbers
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, cluster_df["name"])
                cluster_df["topic cluster"] = topic_cluster_numbers
                st.dataframe(cluster_df[modified_display_columns])
            else:
                st.error("No such useful information was extracted for this repository!")

        with topic_cluster_tab:
            if query_doc.repository_embedding is not None:
                sim_cal_model = load_similaritycal_model("topic")
                cluster_df = run_similaritycal_search(index, repo_topic_clusters, sim_cal_model,
                                                      query_doc, topic_cluster_number, limit)
                topic_cluster_numbers = run_topic_cluster_search(repo_topic_clusters, cluster_df["name"])
                cluster_df["topic cluster"] = topic_cluster_numbers
                code_cluster_numbers = run_code_cluster_search(repo_code_clusters, cluster_df["name"])
                cluster_df["code cluster"] = code_cluster_numbers
                st.dataframe(cluster_df[modified_display_columns])
            else:
                topic_cluster_tab.error("No such useful information was extracted for this repository!")
