from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from chembl_webresource_client.new_client import new_client  # type: ignore
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks import get_openai_callback
from langchain.chat_models.base import BaseChatModel
from langchain.document_loaders import PyPDFLoader
from langchain.tools import tool
from langchain.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.models.molecular_sklearn_regressor import MolecularSKLearnRegressor


def format_docs(docs: list[Document]):
    """Utility function to turn page content of documents into a single string.

    Args:
        docs (list[Document]): List of documents to concatenate

    Returns:
        content (str): Merged content.
    """ """Utility function to turn page content of documents into a single string."""
    content = "\n\n".join(doc.page_content for doc in docs)
    return content


DEFAULT_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "You have access to tools to search attached pdfs and to query "
            "for SMILE strings corresponding to drugs names. "
            "Use your tools when needed for answering the users questions.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@dataclass
class DrugDiscoveryChatBot:
    """
    Chatbot for interrogating papers and looking up drug information.

    Can also perform predictions of the FreeEnergy of a molecule. Use via the chat method.
    Keeps a history of conversations in memory which can be accessed via the get_session_history method.
    The chat method takes a session_id argument to keep the chat_history seperate between sessions.
    A pretrained MolecularSKLearnRegressor for hydration free energy prediction can be attached to the chatbot for use.


    Attributes:
        vectordb_dir (Path | str): The directory path for the vectordb store.
        pdf_path (Optional[Path | str]): The path to the attached PDF file (default: None).
        chunk_size (int): The size of each chunk when splitting the PDF text (default: 1000).
        chunk_overlap (int): The overlap between chunks when splitting the PDF text (default: 100).
        retriever_search_k (int): The number of search results to retrieve (default: 5).
        llm (BaseChatModel): The base chat model for the agent (default: ChatOpenAI(name="gpt-4-turbo")).
        retriever_llm (BaseChatModel): The base chat model for the retriever (default: ChatOpenAI(name="gpt-3.5-turbo")).
        agent_prompt (ChatPromptTemplate): The chat prompt template for the agent (default: DEFAULT_AGENT_PROMPT).
        hydration_free_energy_predictor (Optional[MolecularSKLearnRegressor]): Pretrained model for hydration free
                                                                               energy prediction (default: None).
        verbose_agent (bool): Whether to enable verbose agent logging (default: False).

    Raises:
        TypeError: If the `paper_path` argument is not of type str or Path.
        ValueError: If there is no file at the specified `pdf_file_path`.
        ValueError: If the file at `pdf_file_path` is not a PDF.
    """  # noqa: E501

    vectordb_dir: Path | str
    pdf_path: Optional[Path | str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100
    retriever_search_k: int = 5
    llm: BaseChatModel = field(
        default_factory=lambda: ChatOpenAI(name="gpt-4-turbo")
    )
    retriever_llm: BaseChatModel = field(
        default_factory=lambda: ChatOpenAI(name="gpt-3.5-turbo")
    )
    agent_prompt: ChatPromptTemplate = field(
        default_factory=lambda: DEFAULT_AGENT_PROMPT
    )
    hydration_free_energy_predictor: Optional[MolecularSKLearnRegressor] = None
    verbose_agent: bool = False

    def __post_init__(
        self,
    ) -> None:
        """
        During post_init we attach the pdf file if present
        which initialises the retriever_chain. Then we setup the message
        history database, and finally we initialise the agent executor.
        """
        if self.pdf_path:
            self.attach_pdf_file(self.pdf_path)
        self._message_histories: dict[int, ChatMessageHistory] = {}
        self._agent_executor = self._create_agent_executor()

    def _attach_tools(self) -> None:
        """
        We define and attach the tools to the class here.

        We do this because we want the  tools to have access to class attributes
        and methods.
        However, we don't want to define them as class methods as then they
        need to be called with the self attribute which langchain struggles with.
        """

        @tool
        def query_pdf(
            query: str,
        ) -> str:
            """
            Useful when you need to query information from the attached pdf file.

            The query should be a fully formed question or task.

            Args:
                query (str): The query to be executed on the attached pdf file.

            Returns:
                str: Retriever chain response to the query.

            """
            if self._retriever_chain:
                return self._retriever_chain.invoke(query)
            return (
                "Retriever has not been initialised to an attached pdf file."
            )

        @tool
        def drug_smiles_lookup(
            drug_names: list[str],
        ) -> list[dict[str, str]]:
            """
            Look up smile strings based on drug names in the chembl database.

            Args:
            - drug_names (list[str]): A list of drug names.

            Returns:
            - list[dict[str, str]]: A list of dictionaries representing the found molecules for each drug name input.
              Each dictionary contains the following keys:
              - 'molecule_chembl_id' (str): The ChEMBL ID of the molecule.
              - 'canonical_smiles' (str): The canonical SMILES string of the molecule.
              - 'pref_name' (str): The preferred name of the molecule.
              - 'synonym_name' (str): The drug name used for the lookup.

            """  # noqa: E501
            molecule = new_client.molecule

            results = []
            for drug_name in drug_names:
                mols = molecule.filter(
                    molecule_synonyms__molecule_synonym__iexact=drug_name.strip()
                ).only(
                    "molecule_chembl_id",
                    "pref_name",
                    "molecule_structures",
                )
                results.extend(
                    [
                        {
                            "molecule_chembl_id": mol["molecule_chembl_id"],
                            "canonical_smiles": (
                                mol["molecule_structures"]["canonical_smiles"]
                                if mol["molecule_structures"]
                                else None
                            ),
                            "pref_name": mol["pref_name"],
                            "synonym_name": drug_name,
                        }
                        for mol in mols
                    ]
                )

            return results

        self._tools = [
            query_pdf,
            drug_smiles_lookup,
        ]

        # Define and add predict_hydration_free_energy tool if model available.
        if self.hydration_free_energy_predictor:

            @tool
            def predict_hydration_free_energy(
                smile_str_list: list[str],
            ) -> dict[str, Any]:
                """
                Use to predict the hydration free energy for a list of molecules.
                Takes as input the molecule smile strings in a list.
                Returns a dictionary mapping the smile string to the predicted
                hydration free energy.
                """
                if not self.hydration_free_energy_predictor:
                    raise ValueError(
                        "No hydration free energy predictor attached."
                    )
                prediction_results = (
                    self.hydration_free_energy_predictor.predict(
                        smile_str_list
                    )
                )
                result_dict = {
                    smile: prediction_results[i]
                    for i, smile in enumerate(smile_str_list)
                }
                return result_dict

            self._tools.append(predict_hydration_free_energy)

    def _create_agent_executor(self) -> RunnableWithMessageHistory:
        """
        Create the agent executor for running the agent with tools.

        Tools are initialised inside this function so they have access to the
        class attributes.

        Returns:
            RunnableWithMessageHistory: The agent executor with chat history.
        """
        # Attach tools to class
        self._attach_tools()

        agent = create_tool_calling_agent(
            self.llm, self._tools, self.agent_prompt  # type: ignore[arg-type]
        )

        agent_executor = AgentExecutor(
            agent=agent,  # type: ignore[arg-type]
            tools=self._tools,  # type: ignore[arg-type]
            verbose=self.verbose_agent,
        )

        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,  # type: ignore[arg-type]
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        return agent_with_chat_history

    def _create_retriever_from_pdf(
        self,
        pdf_file_path: Path | str,
    ) -> Runnable:
        """
        Create a retriever chain for querying information from the attached PDF file.

        Uses a Chroma vector store to store the document embeddings.
        Uses self.retriever_llm to rank the documents based on the query.

        Args:
            pdf_file_path (Path | str): The path to the attached PDF file.

        Returns:
            Runnable: The retriever chain.
        """
        if isinstance(
            pdf_file_path,
            str,
        ):
            pdf_file_path = Path(pdf_file_path)
        elif not isinstance(
            pdf_file_path,
            Path,
        ):
            raise TypeError(
                "pdf_file_path argument must be of type str or Path."
            )
        if not pdf_file_path.exists():
            raise ValueError(f"There is no file at {pdf_file_path}.")
        if pdf_file_path.suffix != ".pdf":
            raise ValueError("File is not a pdf.")

        # Create document loader
        pdf_name = pdf_file_path.name.removesuffix(pdf_file_path.suffix)
        embeddings = OpenAIEmbeddings()
        loader = PyPDFLoader(str(pdf_file_path))
        doc = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        )

        # Create vectordb store and initialise retriever
        vectordb_dir_path = (
            Path(self.vectordb_dir)
            if isinstance(self.vectordb_dir, str)
            else self.vectordb_dir
        )
        vectordb_dir_path.mkdir(exist_ok=True, parents=True)
        vector_db = Chroma.from_documents(
            doc,
            embedding=embeddings,
            persist_directory=str(vectordb_dir_path / pdf_name),
        )

        retriever = vector_db.as_retriever(
            search_kwargs={"k": self.retriever_search_k}
        )

        # Create rag_chain runnable
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.retriever_llm
            | StrOutputParser()
        )
        return rag_chain

    def attach_pdf_file(
        self,
        pdf_file_path: Path | str,
    ) -> None:
        """
        Attach a PDF file to the agent for querying information.

        Args:
            pdf_file_path (Path | str): The path to the PDF file.

        Returns:
            None
        """
        self.pdf_path = pdf_file_path
        self._retriever_chain = self._create_retriever_from_pdf(pdf_file_path)

    def get_session_history(self, session_id: int) -> BaseChatMessageHistory:
        """
        Get the chat message history for a specific session ID.

        Args:
            session_id (int): The session ID.

        Returns:
            BaseChatMessageHistory: The chat message history for the specified session ID.
        """

        if session_id not in self._message_histories:
            self._message_histories[session_id] = ChatMessageHistory()
        return self._message_histories[session_id]

    def chat(
        self,
        input: str,
        session_id: int = 0,
    ) -> str:
        """
        Chat with the agent.

        Args:
            input (str): The input message for the agent.
            session_id (int, optional): The session ID for the conversation.
                                        Used to access chat history memory from the session.
                                        Defaults to 0.

        Returns:
            str: The response from the agent.

        """  # noqa: E501
        # Additional logging of cost when using an OpenAI model.
        # TODO: Work out why this seems to only reports the cost of retriever llm.
        opeanai_llm = isinstance(
            self.llm,
            ChatOpenAI,
        ) or isinstance(self.retriever_llm, ChatOpenAI)
        with get_openai_callback() as cb:
            if opeanai_llm:
                config = RunnableConfig(
                    {
                        "configurable": {"session_id": session_id},
                        "callbacks": [cb],
                    }
                )
            else:
                config = RunnableConfig(
                    {
                        "configurable": {"session_id": session_id},
                    }
                )
            try:
                chat_result = self._agent_executor.invoke(
                    input={"input": input},
                    config=config,
                )["output"]
            except Exception as e:
                chat_result = (
                    f"failed to complete agent_chain_run, got error of type "
                    f"{type(e)} with message {e}"
                )
            if opeanai_llm:
                logger.info(f"Total Tokens: {cb.total_tokens}")
                logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
                logger.info(f"Completion Tokens: {cb.completion_tokens}")
                logger.info(f"Successful Requests: {cb.successful_requests}")
                logger.info(f"Total Cost (USD): {cb.total_cost}")
        return chat_result
