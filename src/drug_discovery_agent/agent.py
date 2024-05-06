from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from chembl_webresource_client.new_client import new_client
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models.base import BaseChatModel
from langchain.document_loaders import PyPDFLoader
from langchain.tools import tool
from langchain.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


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
class DrugDiscoveryAgent:
    """_summary_

    Raises:
        TypeError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    vectordb_dir: Path | str
    paper_path: Optional[Path | str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100
    retriever_search_k: int = 5
    llm: BaseChatModel = field(
        default_factory=lambda: ChatOpenAI(name="gpt-4-turbo")
    )
    retriever_llm: BaseChatModel = field(
        default_factory=lambda: ChatOpenAI(name="gpt-3.5-turbo")
    )
    retriever_chain: Optional[RetrievalQA] = field(init=False)
    agent_prompt: BasePromptTemplate = field(
        default_factory=lambda: DEFAULT_AGENT_PROMPT
    )
    verbose_agent: bool = False

    def __post_init__(
        self,
    ):
        if self.paper_path:
            self.attach_pdf_file(self.paper_path)
        self.message_histories: dict[int, ChatMessageHistory] = {}
        self.agent_executor = self._create_agent_executor()

    def get_session_history(self, session_id: int) -> BaseChatMessageHistory:
        if session_id not in self.message_histories:
            self.message_histories[session_id] = ChatMessageHistory()
        return self.message_histories[session_id]

    def _create_agent_executor(self) -> RunnableWithMessageHistory:

        @tool
        def query_pdf(
            query: str,
        ) -> str:
            """
            Useful when you need to query information from the attached pdf file.
            The query should be a fully formed question.
            """
            if self.retriever_chain:
                return self.retriever_chain.invoke(query)
            return (
                "Retriever has not been initialised to an attached pdf file."
            )

        @tool
        def drug_smiles_lookup(
            drug_names: list[str],
        ) -> list[dict[str, str]]:
            """
            This tool can be used to look up smile strings based on drug names in
            the chembl database. The input should be a list of drug names.
            The return will be list of dictionaries for corresponding to
            the found molecules for each drug name input.
            """
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

        tools = [
            query_pdf,
            drug_smiles_lookup,
        ]

        agent = create_tool_calling_agent(self.llm, tools, self.agent_prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=self.verbose_agent
        )

        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
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
        Creates a retriever chain which can be used for querying information from
        the attached pdf file.
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

        pdf_name = pdf_file_path.name.removesuffix(pdf_file_path.suffix)
        embeddings = OpenAIEmbeddings()
        loader = PyPDFLoader(str(pdf_file_path))
        doc = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        )
        vector_db = Chroma.from_documents(
            doc,
            embedding=embeddings,
            persist_directory=str(self.vectordb_dir / pdf_name),
        )

        retriever = vector_db.as_retriever(
            search_kwargs={"k": self.retriever_search_k}
        )

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
        self.retriever_chain = self._create_retriever_from_pdf(pdf_file_path)

    def chat(
        self,
        input: str,
        session_id: int = 0,
    ) -> str:
        # Additional logging of cost when using an OpenAI model.
        # TODO: Work out why this seems to only reports the cost of retriever llm.
        opeanai_llm = isinstance(
            self.llm,
            ChatOpenAI,
        ) or isinstance(self.retriever_llm, ChatOpenAI)
        with get_openai_callback() as cb:
            if opeanai_llm:
                config = {
                    "configurable": {"session_id": session_id},
                    "callbacks": [cb],
                }
            else:
                config = {
                    "configurable": {"session_id": session_id},
                }
            try:
                chat_result = self.agent_executor.invoke(
                    input={"input": input},
                    config=config,
                )
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
