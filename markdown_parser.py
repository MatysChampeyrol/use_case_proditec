"""
Markdown Parser and Preprocessor for RAG Pipeline
Parse and clean converted markdown documents before chunking
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter


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
        """
        Nettoyage avancé optimisé pour le RAG.
        Supprime le bruit de conversion, les numéros de page et reconstruit les phrases.
        """
        
        # 1. Suppression des caractères de contrôle bizarres (sauf newlines/tabs)
        content = "".join(ch for ch in content if ch.isprintable() or ch in '\n\t')

        # 2. Nettoyage ligne par ligne (bruit spécifique et pieds de page)
        lines = content.split('\n')
        cleaned_lines = []
        
        # On utilise _remove_repeating_lines d'abord pour virer les headers/footers récurrents
        lines = self._remove_repeating_lines(lines)

        for line in lines:
            # Supprimer les lignes de points (ex: "..........." ou ". . . .")
            if re.match(r'^\s*[\.\-_]{3,}\s*$', line):
                continue
            
            # Supprimer les patterns de Table des Matières (ex: "Introduction ....... 5")
            # On enlève la partie points + nombre à la fin, mais on garde le titre
            line = re.sub(r'[\.\-_]{3,}\s*\d+\s*$', '', line)
            
            # Supprimer les numéros de page isolés (ex: ligne contenant juste "42" ou "Page 12")
            if re.match(r'^\s*(Page\s*)?\d+\s*$', line, re.IGNORECASE):
                continue
            
            # Supprimer les lignes contenant trop de symboles par rapport au texte (bruit OCR)
            if self._is_noise_line(line):
                continue
                
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)

        # 3. Fusionner les lignes brisées (Reconstruction de phrases)
        # C'est CRITIQUE pour le RAG : transforme "Bonjour je suis\namoon" en "Bonjour je suis amoon"
        content = self._merge_broken_lines(content)

        # 4. Normalisation des espaces et headers
        # Remplacer 3+ retours à la ligne par 2
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Supprimer les espaces multiples dans une ligne
        content = re.sub(r' +', ' ', content)
        # Normaliser les headers (s'assurer qu'il y a un espace après #)
        content = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _is_noise_line(self, line: str) -> bool:
        """Détecte si une ligne est purement du bruit (ex: '___ . ___')"""
        if len(line.strip()) == 0:
            return False
        
        # Compte les caractères alphanumériques
        text_chars = len(re.findall(r'[a-zA-Z0-9à-üÀ-Ü]', line))
        total_chars = len(line.strip())
        
        # Si c'est un header markdown, on garde
        if line.strip().startswith('#'):
            return False
            
        # Si moins de 30% de la ligne est du texte et qu'elle est courte (< 50 chars), c'est souvent du bruit
        if total_chars > 0 and (text_chars / total_chars) < 0.3 and total_chars < 50:
            return True
            
        return False

    def _merge_broken_lines(self, content: str) -> str:
        """
        Fusionne les lignes qui ont été coupées arbitrairement par le format PDF/Word.
        Recolle les mots coupés par un tiret (ex: 'intel- ligence').
        """
        lines = content.split('\n')
        merged_lines = []
        buffer = ""

        for i, line in enumerate(lines):
            line = line.strip()
            
            # Gestion des lignes vides (marquent souvent un vrai paragraphe)
            if not line:
                if buffer:
                    merged_lines.append(buffer)
                    buffer = ""
                merged_lines.append("")
                continue

            # Si c'est un header, une liste ou un tableau, on flush le buffer et on ajoute la ligne telle quelle
            if re.match(r'^(#|\*|-|\+|\|)', line):
                if buffer:
                    merged_lines.append(buffer)
                    buffer = ""
                merged_lines.append(line)
                continue

            # Logique de fusion de texte
            if buffer:
                # Si le buffer finit par un tiret (mot coupé type "ingé- nieur"), on recolle
                if buffer.endswith('-'):
                    # On enlève le tiret et on colle la suite
                    buffer = buffer[:-1] + line
                # Sinon on ajoute un espace standard
                else:
                    buffer += " " + line
            else:
                buffer = line

            # Décision : Doit-on terminer la ligne ici ou continuer à bufferiser ?
            # On arrête de bufferiser si la ligne finit par ., !, ? ou :
            ends_with_punctuation = re.search(r'[.?!:]$', line) is not None
            
            # Lookahead : Regarder la ligne suivante pour voir si c'est un début clair
            next_line_is_start = False
            if i + 1 < len(lines):
                next_l = lines[i+1].strip()
                # Si la prochaine ligne commence par une majuscule (souvent nouvelle phrase)
                # ou est vide, ou est un élément de structure (titre, liste)
                if next_l and (next_l[0].isupper() or re.match(r'^(#|\*|-)', next_l)):
                    next_line_is_start = True
            
            # On flush si on a une ponctuation terminale ou si la suite semble être une nouvelle phrase
            if ends_with_punctuation or next_line_is_start:
                merged_lines.append(buffer)
                buffer = ""

        if buffer:
            merged_lines.append(buffer)

        return '\n'.join(merged_lines)
    
    def _remove_repeating_lines(self, lines: List[str], threshold: int = 3) -> List[str]:
        """Remove lines that appear too frequently (likely headers/footers)"""
        # Count non-empty short lines (< 100 chars, likely headers/footers)
        short_lines = [line.strip() for line in lines if 0 < len(line.strip()) < 100]
        line_counts = Counter(short_lines)
        
        # Find repeating lines (exclude common marks like simple numbers if they aren't page numbers contextually, handled elsewhere)
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
        current_section = {'header': 'Introduction', 'level': 0, 'content': ''}
        
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
        
        # Clean content (New Advanced Logic)
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
        
        # Filter out the failed files list and temporary files
        md_files = [f for f in md_files if not f.name.startswith("_")]
        
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