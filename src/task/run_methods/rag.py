from src.common.parsing import parse_judgement_options
from src.task.build_rag.retriever import BGERetriever, DEFAULT_TOP_K
from src.task.run_methods.base import BaseVariableExtractor
from src.task.run_methods.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


def _format_judgement_options(options: dict) -> str:
    return "\n".join(f"- {key}" for key in options)


def build_retrieval_query(pico: dict, section: dict) -> str:
    return "\n".join(
        [
            f"PICO Question: {pico.get('Question', '')}",
            f"Population: {pico.get('Population', '')}",
            f"Intervention: {pico.get('Intervention', '')}",
            f"Comparison: {pico.get('Comparison', '')}",
            f"Outcomes: {pico.get('Main outcomes', '')}",
            f"EtD criterion: {section.get('criterion', '')}",
            f"EtD subquestion: {section.get('question', '')}",
        ]
    )


class RagVariableExtractor(BaseVariableExtractor):
    def __init__(
        self,
        retriever: BGERetriever,
        *,
        top_k: int = DEFAULT_TOP_K,
        exclude_self_pico: bool = False,
    ) -> None:
        """
        Args:
            exclude_self_pico:
                If True, exclude chunks whose source_file matches the current
                PICO file (leave-one-PICO-out mode; simulates deployment where
                the reference evidence is not yet in the knowledge base).
                If False (default, evaluation mode), retrieve from the full
                corpus including the reference PICO's own evidence — used when the
                goal is to score whether the model can locate the correct
                evidence among all available evidence.
        """
        self.retriever = retriever
        self.top_k = top_k
        self.exclude_self_pico = exclude_self_pico

    def extract(self, etd_data, item, idx):
        pico = item["pico"]
        section = item["sections"][idx]

        judgement_options = parse_judgement_options(section["judgement_extract"])
        query = build_retrieval_query(pico, section)
        retrieved_text, retrieved_chunks = self.retriever.retrieve_formatted(
            query,
            top_k=self.top_k,
            exclude_source_file=item["source_file"] if self.exclude_self_pico else None,
        )

        system_prompt = RAG_SYSTEM_PROMPT.substitute(
            POPULATION=pico["Population"],
            INTERVENTION=pico["Intervention"],
            COMPARISON=pico["Comparison"],
            MAIN_OUTCOMES=pico["Main outcomes"],
        )

        user_prompt = RAG_USER_PROMPT.substitute(
            CRITERION=section["criterion"],
            SUBQUESTION=section["question"],
            RESEARCH_EVIDENCE=retrieved_text or "No relevant evidence retrieved.",
            JUDGEMENT_OPTIONS=_format_judgement_options(judgement_options),
        )

        subquestion_dict = {
            "Population": pico["Population"],
            "Intervention": pico["Intervention"],
            "Comparison": pico["Comparison"],
            "Main outcomes": pico["Main outcomes"],
            "criterion": section["criterion"],
            "question": section["question"],
            "gt_judgement": judgement_options,
            "retrieved_evidence": retrieved_text,
            "retrieved_chunks": retrieved_chunks,
        }

        return system_prompt, user_prompt, subquestion_dict
