"""
Document Conversion Script - Convert documents to Markdown using markitdown
Supported formats: .docx, .pdf, .doc, .xlsx, .pptx, .xls
For old .doc files that fail, uses Microsoft Word COM interface as fallback
"""

import sys
import os
import shutil
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from markitdown import MarkItDown

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # tqdm is optional

# Configuration
SOURCE_DIR = Path("proditec/")  # Source directory
OUTPUT_DIR = Path("converted_docs")  # Output directory
TEMP_DIR = Path("temp_docx")  # Temporary folder for converted .docx files
FAILED_FILES_PATH = OUTPUT_DIR / "_failed_files.txt"

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.docx', '.pdf', '.doc', '.xlsx', '.pptx', '.xls'}


def _safe_write_text(output_path: Path, content: str) -> None:
    """
    Write text to a temp file then atomically replace the target file.
    Helps avoid partial outputs when the process is interrupted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.replace(tmp_path, output_path)


def is_output_complete(input_path: Path, output_path: Path) -> bool:
    """
    Determine whether an output Markdown file is complete enough to skip.
    Criteria:
      - output exists
      - output is non-empty
      - output mtime >= input mtime (best-effort)
    """
    try:
        if not output_path.exists():
            return False
        if output_path.stat().st_size <= 0:
            return False
        return output_path.stat().st_mtime >= input_path.stat().st_mtime
    except Exception:
        return False


def convert_file(md_converter: MarkItDown, input_path: Path, output_path: Path) -> tuple[bool, str | None]:
    """
    Convert a single file to Markdown format
    
    Args:
        md_converter: MarkItDown converter instance
        input_path: Input file path
        output_path: Output file path
    
    Returns:
        (bool, error): Whether conversion was successful and optional error message
    """
    try:
        # Convert file
        result = md_converter.convert(str(input_path))
        
        # Write converted content safely
        _safe_write_text(output_path, result.text_content)

        return True, None
    except Exception as e:
        return False, str(e)


def get_all_documents(source_dir: Path) -> list[Path]:
    """
    Recursively get all supported document files
    
    Args:
        source_dir: Source directory
    
    Returns:
        list: List of document file paths
    """
    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        documents.extend(source_dir.rglob(f"*{ext}"))
    return sorted(documents)


def get_failed_doc_files() -> list[Path]:
    """Read the list of failed files and return .doc/.docx files"""
    failed_files = []
    
    if not FAILED_FILES_PATH.exists():
        print(f"[ERROR] Failed files list not found: {FAILED_FILES_PATH}")
        return failed_files
    
    with open(FAILED_FILES_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header lines
    for line in lines[2:]:  # Skip "List of failed files:" and separator
        line = line.strip()
        if line and (line.lower().endswith('.doc') or line.lower().endswith('.docx')):
            file_path = SOURCE_DIR / line
            if file_path.exists():
                failed_files.append(file_path)
    
    return failed_files


def convert_doc_to_docx(word_app, doc_path: Path, docx_path: Path) -> bool:
    """
    Convert .doc to .docx using Word COM interface
    
    Args:
        word_app: Word application COM object
        doc_path: Input .doc file path
        docx_path: Output .docx file path
    
    Returns:
        bool: Whether conversion was successful
    """
    try:
        # Ensure output directory exists
        docx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open the .doc file
        doc = word_app.Documents.Open(str(doc_path.absolute()))
        
        # Save as .docx (format 16 = docx)
        doc.SaveAs2(str(docx_path.absolute()), FileFormat=16)
        doc.Close()
        
        return True
    except Exception as e:
        print(f"  [ERROR] Word conversion failed: {e}")
        return False


def convert_docx_to_md(md_converter: MarkItDown, docx_path: Path, md_path: Path) -> bool:
    """
    Convert .docx to Markdown
    
    Args:
        md_converter: MarkItDown converter instance
        docx_path: Input .docx file path
        md_path: Output .md file path
    
    Returns:
        bool: Whether conversion was successful
    """
    try:
        result = md_converter.convert(str(docx_path))
        
        _safe_write_text(md_path, result.text_content)
        
        return True
    except Exception as e:
        print(f"  [ERROR] Markdown conversion failed: {e}")
        return False


_thread_local = threading.local()


def get_thread_local_converter() -> MarkItDown:
    converter = getattr(_thread_local, "converter", None)
    if converter is None:
        converter = MarkItDown()
        _thread_local.converter = converter
    return converter


def write_failed_files(all_documents: list[Path]) -> int:
    """
    Write/update the failed files list based on missing/incomplete outputs.
    This makes the process resumable across runs.
    """
    missing = []
    for doc_path in all_documents:
        relative_path = doc_path.relative_to(SOURCE_DIR)
        md_path = OUTPUT_DIR / relative_path.with_suffix('.md')
        if not is_output_complete(doc_path, md_path):
            missing.append(str(relative_path))

    if missing:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(FAILED_FILES_PATH, 'w', encoding='utf-8') as f:
            f.write("List of failed files:\n")
            f.write("-" * 40 + "\n")
            for file in missing:
                f.write(f"{file}\n")
        print(f"\nFailed files list saved to: {FAILED_FILES_PATH}")
    else:
        if FAILED_FILES_PATH.exists():
            FAILED_FILES_PATH.unlink()
        print("\nAll files converted successfully! Removed failed files list.")

    return len(missing)


def run_main_conversion(workers: int, resume: bool, overwrite: bool, show_progress: bool, verbose: bool):
    """Main conversion function - convert all documents using markitdown"""
    print("=" * 60)
    print("Document Conversion Tool - Using MarkItDown")
    print("=" * 60)
    
    # Check if source directory exists
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all documents
    documents = get_all_documents(SOURCE_DIR)
    total_files = len(documents)
    
    print(f"\nFound {total_files} document files to convert")
    print(f"Source directory: {SOURCE_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Workers: {workers}")
    print(f"Resume: {resume} | Overwrite: {overwrite}")
    print("-" * 60)

    if show_progress and tqdm is None:
        print("[WARN] tqdm is not installed, progress bar disabled. Install with: pip install tqdm")
        show_progress = False

    # Statistics
    success_count = 0
    fail_count = 0
    skip_count = 0

    bar = None
    if show_progress and tqdm is not None:
        bar = tqdm(total=total_files, desc="Converting", unit="file")

    def _update_bar(n: int = 1) -> None:
        if bar is not None:
            bar.update(n)

    # Prepare tasks
    tasks = []
    for doc_path in documents:
        relative_path = doc_path.relative_to(SOURCE_DIR)
        output_path = OUTPUT_DIR / relative_path.with_suffix('.md')

        if not overwrite and resume and is_output_complete(doc_path, output_path):
            skip_count += 1
            if verbose and not show_progress:
                print(f"[SKIP] {relative_path}")
            _update_bar(1)
            continue

        tasks.append((doc_path, output_path, relative_path))

    if verbose:
        print(f"Total: {total_files} | To convert: {len(tasks)} | Skipped: {skip_count}")

    # Convert with concurrency
    if workers <= 1:
        for doc_path, output_path, relative_path in tasks:
            if verbose and not show_progress:
                print(f"[CONVERT] {relative_path}")
            ok, err = convert_file(get_thread_local_converter(), doc_path, output_path)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                if verbose and err:
                    print(f"  [ERROR] {relative_path}: {err}")
            _update_bar(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(convert_file, get_thread_local_converter(), doc_path, output_path): (doc_path, output_path, relative_path)
                for (doc_path, output_path, relative_path) in tasks
            }
            for fut in as_completed(future_map):
                doc_path, output_path, relative_path = future_map[fut]
                try:
                    ok, err = fut.result()
                except Exception as e:
                    ok, err = False, str(e)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    if verbose and err:
                        print(f"  [ERROR] {relative_path}: {err}")
                _update_bar(1)

    if bar is not None:
        bar.close()

    # Output statistics
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"  Success: {success_count} files")
    print(f"  Failed: {fail_count} files")
    print(f"  Skipped: {skip_count} files")
    print("=" * 60)

    remaining = write_failed_files(documents)
    if remaining > 0:
        print("Run with --retry-doc to retry .doc files using Word COM interface")

    return remaining


def run_doc_retry(resume: bool, overwrite: bool, show_progress: bool, verbose: bool):
    """Retry failed .doc files using Word COM interface"""
    print("=" * 60)
    print("Retry Failed .doc Files Using Word COM Interface")
    print("=" * 60)
    
    # Import win32com only when needed (Windows only)
    try:
        import win32com.client
    except ImportError:
        print("[ERROR] win32com is not installed. Install it with: pip install pywin32")
        sys.exit(1)
    
    # Get failed .doc files
    failed_files = get_failed_doc_files()
    total_files = len(failed_files)
    
    if total_files == 0:
        print("\nNo .doc files to convert.")
        return
    
    print(f"\nFound {total_files} .doc files to convert")
    print("-" * 60)

    if show_progress and tqdm is None:
        print("[WARN] tqdm is not installed, progress bar disabled. Install with: pip install tqdm")
        show_progress = False
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Word application
    print("\nStarting Microsoft Word...")
    try:
        word_app = win32com.client.Dispatch("Word.Application")
        word_app.Visible = False  # Run in background
    except Exception as e:
        print(f"[ERROR] Could not start Microsoft Word: {e}")
        print("Please ensure Microsoft Word is installed.")
        return
    
    # Initialize markitdown
    md_converter = MarkItDown()
    
    # Statistics
    success_count = 0
    fail_count = 0
    still_failed = []

    bar = None
    if show_progress and tqdm is not None:
        bar = tqdm(total=total_files, desc="Retry .doc", unit="file")

    try:
        for idx, doc_path in enumerate(failed_files, 1):
            relative_path = doc_path.relative_to(SOURCE_DIR)

            md_path = OUTPUT_DIR / relative_path.with_suffix('.md')
            if not overwrite and resume and is_output_complete(doc_path, md_path):
                if verbose and not show_progress:
                    print(f"[SKIP] {relative_path}")
                if bar is not None:
                    bar.update(1)
                continue

            if verbose and not show_progress:
                print(f"[PROCESS] {relative_path}")
            
            # Paths
            docx_path = TEMP_DIR / relative_path.with_suffix('.docx')
            
            # Step 1: Convert .doc to .docx
            if verbose and not show_progress:
                print("  Converting to .docx...")
            if not convert_doc_to_docx(word_app, doc_path, docx_path):
                fail_count += 1
                still_failed.append(str(relative_path))
                if bar is not None:
                    bar.update(1)
                continue
            
            # Step 2: Convert .docx to .md
            if verbose and not show_progress:
                print("  Converting to .md...")
            if convert_docx_to_md(md_converter, docx_path, md_path):
                success_count += 1
                if verbose and not show_progress:
                    print(f"  [OK] -> {md_path}")
            else:
                fail_count += 1
                still_failed.append(str(relative_path))

            if bar is not None:
                bar.update(1)
    
    finally:
        # Close Word
        print("\nClosing Microsoft Word...")
        word_app.Quit()
        if bar is not None:
            bar.close()
    
    # Output statistics
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"  Success: {success_count} files")
    print(f"  Failed: {fail_count} files")
    print("=" * 60)
    
    # Update failed files list
    if still_failed:
        with open(FAILED_FILES_PATH, 'w', encoding='utf-8') as f:
            f.write("List of failed files:\n")
            f.write("-" * 40 + "\n")
            for file in still_failed:
                f.write(f"{file}\n")
        print(f"\nUpdated failed files list: {FAILED_FILES_PATH}")
    else:
        # All files converted, remove the failed files list
        if FAILED_FILES_PATH.exists():
            FAILED_FILES_PATH.unlink()
        print("\nAll files converted successfully! Removed failed files list.")
    
    # Clean up temp directory
    print(f"\nCleaning up temporary files...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("Done!")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Convert documents to Markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_docs.py              # Convert all documents
  python convert_docs.py --retry-doc  # Retry failed .doc files using Word
  python convert_docs.py --workers 8  # Convert with concurrency
        """
    )
    
    parser.add_argument(
        '--retry-doc',
        action='store_true',
        help='Retry failed .doc files using Microsoft Word COM interface'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Source directory path (default: prodicet)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory path (default: converted_docs)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help='Number of worker threads for conversion (default: min(8, cpu_count))'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume mode (do not skip existing outputs)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite outputs even if they already exist'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print more logs'
    )
    
    args = parser.parse_args()
    
    # Update paths if provided
    global SOURCE_DIR, OUTPUT_DIR, FAILED_FILES_PATH
    if args.source:
        SOURCE_DIR = Path(args.source)
    if args.output:
        OUTPUT_DIR = Path(args.output)
        FAILED_FILES_PATH = OUTPUT_DIR / "_failed_files.txt"
    
    resume = not args.no_resume
    overwrite = bool(args.overwrite)
    show_progress = not args.no_progress
    verbose = bool(args.verbose)

    if args.retry_doc:
        run_doc_retry(resume=resume, overwrite=overwrite, show_progress=show_progress, verbose=verbose)
    else:
        run_main_conversion(workers=max(1, args.workers), resume=resume, overwrite=overwrite, show_progress=show_progress, verbose=verbose)


if __name__ == "__main__":
    main()
