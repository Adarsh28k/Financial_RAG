import sys
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import SectionHeaderItem, TableItem, PictureItem, TextItem

PDF_SUBDIR = "pdf"
MARKDOWN_SUBDIR = "markdown"
IMAGES_SUBDIR = "images"


def build_converter() -> DocumentConverter:
    """
    A plain DocumentConverter() will NOT keep picture bytes around -
    PictureItem.image stays None unless we ask for it explicitly.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_pdf_to_markdown(pdf_path: str, images_dir: Path) -> tuple[str, int]:
    """
    Convert PDF to Markdown using Docling's native structure detection.
    """
    converter = build_converter()
    result = converter.convert(pdf_path)
    document = result.document

    pdf_name = Path(pdf_path).stem
    image_count = 0
    lines = []

    for item, _level in document.iterate_items():

        if isinstance(item, SectionHeaderItem):
            level = min(item.level, 6)
            lines.append(f"{'#' * level} {item.text}\n\n")

        elif isinstance(item, TableItem):
            # Docling already knows how to render its own tables correctly,
            # including merged cells - no need to walk cells by hand.
            lines.append(item.export_to_markdown(document))
            lines.append("\n\n")

        elif isinstance(item, PictureItem):
            if item.image is None:
                continue  # no image bytes captured for this figure, skip it

            image_filename = f"{pdf_name}_image_{image_count:03d}.png"
            image_path = images_dir / image_filename

            try:
                item.image.pil_image.save(image_path)
            except OSError as e:
                print(f"Warning: could not save {image_filename}: {e}")
                continue

            caption = item.caption_text(doc=document)
            alt_text = caption if caption else f"Figure from {pdf_name}"
            lines.append(f"![{alt_text}](../{IMAGES_SUBDIR}/{image_filename})\n\n")
            if caption:
                lines.append(f"*{caption}*\n\n")

            image_count += 1

        elif isinstance(item, TextItem):
            lines.append(f"{item.text}\n\n")

    return "".join(lines), image_count


def process_folder(folder_path: str):
    """Process all PDFs in a folder."""
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    pdf_dir = folder / PDF_SUBDIR
    markdown_dir = folder / MARKDOWN_SUBDIR
    images_dir = folder / IMAGES_SUBDIR
    markdown_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{pdf_dir}'\n")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        markdown_content, image_count = extract_pdf_to_markdown(str(pdf_file), images_dir)

        output_file = markdown_dir / (pdf_file.stem + ".md")
        output_file.write_text(markdown_content, encoding="utf-8")

        print(f"  Markdown saved: {output_file.name}")
        print(f"  Images extracted: {image_count}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_markdown.py <folder_path>")
        sys.exit(1)

    process_folder(sys.argv[1])