import os
import re
from pymupdf4llm import to_markdown
from langchain_text_splitters import MarkdownHeaderTextSplitter

class PDFService:
    def __init__(self):
        self.headers_to_split = [
            ("##", "section")
        ]
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split)

    def to_markdown(self, pdf_path: str) -> str:
        return to_markdown(pdf_path)

    def split_and_clean(self, md_text: str) -> list:
        # 1. Split
        chunks = self.splitter.split_text(md_text)
        
        # 2. Propagate section names
        last_section = ""
        for chunk in chunks:
            if chunk.metadata.get("section", "") != "":
                last_section = chunk.metadata["section"]
            else:
                chunk.metadata["section"] = last_section
        
        # 3. Clean and Filter
        cleaned_chunks = []
        for chunk in chunks:
            content = chunk.page_content
            # Remove picture placeholder lines
            content = re.sub(r'\*\*==> picture.*?<==\*\*', '', content)
            content = re.sub(r'\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*', '', content, flags=re.DOTALL)
            content = content.strip()
            
            # Drop if too short (10 chars)
            if len(content) < 10:
                continue
            
            chunk.page_content = content
            cleaned_chunks.append(chunk)
            
        return cleaned_chunks
