# src/extractors/base.py
class BaseVariableExtractor:
    def extract(self, etd_data, item, idx):
        raise NotImplementedError
