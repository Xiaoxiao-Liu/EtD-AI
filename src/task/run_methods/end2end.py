from src.common.parsing import parse_judgement_options
from src.task.run_methods.base import BaseVariableExtractor
from src.task.run_methods.prompts import END2END_SYSTEM_PROMPT, END2END_USER_PROMPT


class End2EndVariableExtractor(BaseVariableExtractor):

    def extract(self, etd_data, item, idx):
        pico = item["pico"]
        section = item["sections"][idx]

        judgement_options = parse_judgement_options(section["judgement_extract"])

        system_prompt = END2END_SYSTEM_PROMPT.substitute(
            POPULATION=pico["Population"],
            INTERVENTION=pico["Intervention"],
            COMPARISON=pico["Comparison"],
            MAIN_OUTCOMES=pico["Main outcomes"],
        )

        user_prompt = END2END_USER_PROMPT.substitute(
            CRITERION=section["criterion"],
            SUBQUESTION=section["question"],
            JUDGEMENT_OPTIONS=judgement_options.keys(),
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
