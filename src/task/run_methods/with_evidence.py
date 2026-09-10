from src.common.parsing import parse_judgement_options
from src.task.run_methods.base import BaseVariableExtractor
from src.task.run_methods.prompts import WITH_EVIDENCE_SYSTEM_PROMPT, WITH_EVIDENCE_USER_PROMPT


class WithEvidenceVariableExtractor(BaseVariableExtractor):

    def extract(self, etd_data, item, idx):
        pico = item["pico"]
        section = item["sections"][idx]

        judgement_options = parse_judgement_options(section["judgement_extract"])

        system_prompt = WITH_EVIDENCE_SYSTEM_PROMPT.substitute(
            POPULATION=pico["Population"],
            INTERVENTION=pico["Intervention"],
            COMPARISON=pico["Comparison"],
            MAIN_OUTCOMES=pico["Main outcomes"],
        )

        user_prompt = WITH_EVIDENCE_USER_PROMPT.substitute(
            CRITERION=section["criterion"],
            SUBQUESTION=section["question"],
            RESEARCH_EVIDENCE=section["research_evidence"],
            JUDGEMENT_OPTIONS=judgement_options,
        )

        subquestion_dict = {
            "Population": pico["Population"],
            "Intervention": pico["Intervention"],
            "Comparison": pico["Comparison"],
            "Main outcomes": pico["Main outcomes"],
            "criterion": section["criterion"],
            "question": section["question"],
            "gt_judgement": judgement_options,
        }

        return system_prompt, user_prompt, subquestion_dict
