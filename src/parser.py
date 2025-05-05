
"""
Input: PDF document path
Output: Text and table data extracted into a Markdown (.md) file
PDF Parsing Tool: MinerU (via Magic-PDF)
"""
import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def parse_pdf_to_md(pdf_path: str) -> None:
    """Extracts content from PDF and writes structured markdown and JSON files."""
    name_without_ext = os.path.basename(pdf_path).split('.')[0]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    output_img_dir = os.path.join(base_dir, "outputs", name_without_ext, "images")
    output_md_dir = os.path.join(base_dir, "outputs", name_without_ext)
    image_dir_name = os.path.basename(output_img_dir)

    os.makedirs(output_img_dir, exist_ok=True)

    image_writer = FileBasedDataWriter(output_img_dir)
    md_writer = FileBasedDataWriter(output_md_dir)

    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)

    dataset = PymuDocDataset(pdf_bytes)

    if dataset.classify() == SupportedPdfParseMethod.OCR:
        result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = result.pipe_ocr_mode(image_writer)
    else:
        result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = result.pipe_txt_mode(image_writer)

    result.get_infer_res()
    pipe_result.draw_layout(os.path.join(output_md_dir, f"{name_without_ext}_layout.pdf"))
    pipe_result.draw_span(os.path.join(output_md_dir, f"{name_without_ext}_spans.pdf"))

    pipe_result.get_markdown(image_dir_name)
    pipe_result.dump_md(md_writer, f"{name_without_ext}.md", image_dir_name)

    pipe_result.get_content_list(image_dir_name)
    pipe_result.dump_content_list(md_writer, f"{name_without_ext}_content_list.json", image_dir_name)

    pipe_result.get_middle_json()
    pipe_result.dump_middle_json(md_writer, f"{name_without_ext}_middle.json")

if __name__ == "__main__":
    SAMPLE_PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples/Attention_arxiv.pdf")
    parse_pdf_to_md(SAMPLE_PDF_PATH)
