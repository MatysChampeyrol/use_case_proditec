"""
Markdown Parser and Preprocessor for RAG Pipeline
Parse and clean converted markdown documents before chunking
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class DocumentChunk:
    """Represents a parsed document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class MarkdownParser:
    """Parse and preprocess markdown documents for RAG"""
    
    def __init__(self, input_dir: str = "converted_docs", output_dir: str = "parsed_docs"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_markdown(self, content: str) -> str:
        """Clean and normalize markdown content"""
        
        # Remove excessive blank lines (3+ → 2)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove excessive spaces
        content = re.sub(r' +', ' ', content)
        
        # Remove trailing spaces
        content = re.sub(r' +\n', '\n', content)
        
        # Normalize section headers (ensure space after #)
        content = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', content, flags=re.MULTILINE)
        
        # Remove page numbers patterns (common in PDFs)
        content = re.sub(r'^\s*Page \d+\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r'^\s*\d+\s*/\s*\d+\s*$', '', content, flags=re.MULTILINE)
        
        # Remove header/footer artifacts (lines that repeat)
        lines = content.split('\n')
        cleaned_lines = self._remove_repeating_lines(lines)
        content = '\n'.join(cleaned_lines)
        
        return content.strip()
    
    def _remove_repeating_lines(self, lines: List[str], threshold: int = 3) -> List[str]:
        """Remove lines that appear too frequently (likely headers/footers)"""
        from collections import Counter
        
        # Count non-empty short lines (< 100 chars, likely headers/footers)
        short_lines = [line.strip() for line in lines if 0 < len(line.strip()) < 100]
        line_counts = Counter(short_lines)
        
        # Find repeating lines
        repeating = {line for line, count in line_counts.items() if count >= threshold}
        
        # Filter out repeating lines
        return [line for line in lines if line.strip() not in repeating]
    
    def extract_metadata(self, content: str, filepath: Path) -> Dict[str, Any]:
        """Extract metadata from document"""
        
        metadata = {
            'source_file': filepath.name,
            'source_path': str(filepath.relative_to(self.input_dir)),
            'file_type': filepath.suffix,
        }
        
        # Extract title (first H1 or filename)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        else:
            metadata['title'] = filepath.stem
        
        # Extract all section headers
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata['sections'] = [{'level': len(h[0]), 'title': h[1].strip()} for h in headers]
        metadata['num_sections'] = len(headers)
        
        # Document statistics
        metadata['char_count'] = len(content)
        metadata['word_count'] = len(content.split())
        metadata['line_count'] = len(content.split('\n'))
        
        return metadata
    
    def split_by_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split document by sections (headers)"""
        
        # Split on headers while keeping them
        sections = []
        current_section = {'header': '', 'level': 0, 'content': ''}
        
        for line in content.split('\n'):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    'header': header_match.group(2).strip(),
                    'level': len(header_match.group(1)),
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def parse_document(self, filepath: Path) -> Dict[str, Any]:
        """Parse a single markdown document"""
        
        print(f"Parsing: {filepath.name}")
        
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Clean content
        cleaned_content = self.clean_markdown(raw_content)
        
        # Extract metadata
        metadata = self.extract_metadata(cleaned_content, filepath)
        
        # Split by sections
        sections = self.split_by_sections(cleaned_content)
        
        return {
            'metadata': metadata,
            'cleaned_content': cleaned_content,
            'sections': sections,
            'raw_content': raw_content
        }
    
    def parse_all(self, save_json: bool = True, save_cleaned: bool = True) -> List[Dict[str, Any]]:
        """Parse all markdown files in input directory"""
        
        print("=" * 60)
        print("Markdown Parser - Preprocessing for RAG")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        # Find all markdown files
        md_files = list(self.input_dir.rglob("*.md"))
        
        # Filter out the failed files list
        md_files = [f for f in md_files if f.name != "_failed_files.txt"]
        
        if not md_files:
            print("No markdown files found!")
            return []
        
        print(f"Found {len(md_files)} markdown files\n")
        
        parsed_docs = []
        
        for md_file in md_files:
            try:
                parsed = self.parse_document(md_file)
                parsed_docs.append(parsed)
                
                # Save cleaned markdown
                if save_cleaned:
                    relative_path = md_file.relative_to(self.input_dir)
                    output_path = self.output_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(parsed['cleaned_content'])
                
                # Save metadata JSON
                if save_json:
                    json_path = self.output_dir / md_file.relative_to(self.input_dir).with_suffix('.json')
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'metadata': parsed['metadata'],
                            'sections': parsed['sections']
                        }, f, indent=2, ensure_ascii=False)
                
                # Print summary
                print(f"✓ {md_file.name}")
                print(f"  Title: {parsed['metadata']['title']}")
                print(f"  Sections: {parsed['metadata']['num_sections']}")
                print(f"  Words: {parsed['metadata']['word_count']:,}")
                print()
                
            except Exception as e:
                print(f"✗ Error parsing {md_file.name}: {e}\n")
                continue
        
        print("=" * 60)
        print(f"Parsing Complete! Processed {len(parsed_docs)} documents")
        print(f"Cleaned markdown saved to: {self.output_dir}")
        print("=" * 60)
        
        return parsed_docs


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse and preprocess markdown documents")
    parser.add_argument('--input', type=str, default='converted_docs', help='Input directory')
    parser.add_argument('--output', type=str, default='parsed_docs', help='Output directory')
    parser.add_argument('--no-json', action='store_true', help='Skip saving JSON metadata')
    parser.add_argument('--no-cleaned', action='store_true', help='Skip saving cleaned markdown')
    
    args = parser.parse_args()
    
    parser_obj = MarkdownParser(input_dir=args.input, output_dir=args.output)
    parsed_docs = parser_obj.parse_all(
        save_json=not args.no_json,
        save_cleaned=not args.no_cleaned
    )
    
    print(f"\nTotal documents parsed: {len(parsed_docs)}")


if __name__ == "__main__":
    main()
