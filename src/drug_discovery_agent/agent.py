from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from chembl_webresource_client.new_client import new_client
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models.base import BaseChatModel
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.tools import tool
from langchain.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger


@dataclass
class DrugDiscoveryAgent:
    paper_path: Optional[Path | str]
    vectordb_dir: Optional[Path | str]
    chunk_size: int = 500
    chunk_overlap: int = 0
    llm: BaseChatModel = field(
        default_factory=lambda: ChatOpenAI(name="gpt-4-turbo")
    )
    retriever_llm: BaseChatModel = field(
        default_factory=lambda: ChatOpenAI(name="gpt-3.5-turbo")
    )
    retriever: Optional[RetrievalQA] = field(init=False)
    memory: BaseChatMemory = field(
        default_factory=lambda: ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    )
    verbose_agent: bool = False

    def __post_init__(
        self,
    ):
        if self.paper_path:
            self.attach_pdf_file(self.paper_path)

        @tool
        def query_pdf(
            query: str,
        ) -> str:
            """
            Useful when you need to query information from the attached pdf file.
            The query should be a fully formed question.
            """
            if self.retriever:
                return self.retriever.run(query)
            return (
                "Retriever has not been initialised to an attached pdf file."
            )

        @tool
        def drug_smiles_lookup(
            query: str,
        ) -> str:
            """
            This tool can be used to look up smile strings based on drug names in
            the chembl database. The input should be a comma seperated list of
            drug names. The return will be a comma seperated list of dictionaries for
            each drug name input, with different drug names separated by
            "\n----\n" characters.
            """
            molecule = new_client.molecule
            drug_names = query.split(",")

            result_str = ""
            for drug_name in drug_names:
                mols = molecule.filter(
                    molecule_synonyms__molecule_synonym__iexact=drug_name.strip()
                ).only(
                    "molecule_chembl_id",
                    "pref_name",
                    "molecule_structures",
                )
                records = [
                    {
                        "molecule_chembl_id": mol["molecule_chembl_id"],
                        "canonical_smiles": mol["molecule_structures"][
                            "canonical_smiles"
                        ],
                        "pref_name": mol["pref_name"],
                        "synonym_name": drug_name,
                    }
                    for mol in mols
                ]
                result_str += ",".join([str(record) for record in records])
                result_str += "\n----\n"
            return result_str

        tools = [
            query_pdf,
            drug_smiles_lookup,
        ]

        self.agent_chain = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=self.verbose_agent,
        )

    def _create_retriever_from_pdf(
        self,
        pdf_file_path: Path | str,
    ) -> RetrievalQA:
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
            text_splitter=CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        )
        vector_db = Chroma.from_documents(
            doc,
            embedding=embeddings,
            persist_directory=str(self.vectordb_dir / pdf_name),
        )
        retriever = RetrievalQA.from_chain_type(
            llm=self.retriever_llm,
            chain_type="refine",
            retriever=vector_db.as_retriever(),
        )
        return retriever

    def attach_pdf_file(
        self,
        pdf_file_path: Path | str,
    ) -> None:
        self.retriever = self._create_retriever_from_pdf(pdf_file_path)

    def chat(
        self,
        input: str,
    ) -> str:
        # Additional logging of cost when using an OpenAI model.
        if isinstance(
            self.llm,
            ChatOpenAI,
        ):
            with get_openai_callback() as cb:
                try:
                    chat_result = self.agent_chain.invoke(input=input)
                except Exception as e:
                    chat_result = (
                        f"failed to complete agent_chain_run, got error of type "
                        f"{type(e)} with message {e}"
                    )
                logger.info(f"Total Tokens: {cb.total_tokens}")
                logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
                logger.info(f"Completion Tokens: {cb.completion_tokens}")
                logger.info(f"Successful Requests: {cb.successful_requests}")
                logger.info(f"Total Cost (USD): {cb.total_cost}")
                return chat_result
        return self.agent_chain.invoke(input=input)
