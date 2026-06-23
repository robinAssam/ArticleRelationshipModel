"""
Generate an Obsidian Vault from atom annotations, incorporating semantic
similarities and IDF weights for enriched graph interconnections.
"""
import os
import sys
from pathlib import Path
import yaml
import shutil
import re

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.atom_table import (
    build_atom_table,
    compute_tag_weights,
    compute_tag_similarities,
    LIST_FIELDS,
    unique_tags_by_field
)

VAULT_DIR = PROJECT_ROOT / "data" / "obsidian_vault"

def sanitize_filename(name: str) -> str:
    """Sanitize strings for Windows filenames."""
    # Replace invalid Windows filename characters
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def main():
    print(f"Building Obsidian Vault at {VAULT_DIR}...")
    
    # 1. Load Data
    df = build_atom_table(PROJECT_ROOT / "data" / "annotations")
    print(f"Loaded {len(df)} atoms.")
    
    # 2. Compute NLP attributes
    print("Computing IDF weights...")
    weights = compute_tag_weights(df, method="idf")
    
    print("Computing semantic similarities (this may download a model)...")
    sim_pairs = compute_tag_similarities(df, threshold=0.55)
    
    # 3. Create structure
    if VAULT_DIR.exists():
        shutil.rmtree(VAULT_DIR)
        
    (VAULT_DIR / "Articles").mkdir(parents=True, exist_ok=True)
    (VAULT_DIR / "Atoms").mkdir(parents=True, exist_ok=True)
    (VAULT_DIR / "Entities").mkdir(parents=True, exist_ok=True)
    
    # Map for semantic links
    semantic_map = {}
    for _, row in sim_pairs.iterrows():
        field = row["field"]
        tag_a = row["tag_a"]
        tag_b = row["tag_b"]
        sim = row["similarity"]
        
        if field not in semantic_map:
            semantic_map[field] = {}
        
        if tag_a not in semantic_map[field]:
            semantic_map[field][tag_a] = []
        semantic_map[field][tag_a].append((tag_b, sim))
        
        if tag_b not in semantic_map[field]:
            semantic_map[field][tag_b] = []
        semantic_map[field][tag_b].append((tag_a, sim))

    # Generate Entities
    print("Generating Entities...")
    tags_by_field = unique_tags_by_field(df)
    for field, tags in tags_by_field.items():
        for tag in tags:
            weight = weights.get((field, tag), 0.0)
            
            frontmatter = {
                "type": "entity",
                "field": field,
                "weight": round(weight, 4)
            }
            
            content = [
                "---",
                yaml.dump(frontmatter, sort_keys=False).strip(),
                "---",
                f"# Entity: {tag}",
                ""
            ]
            
            # Add semantic links
            similar = semantic_map.get(field, {}).get(tag, [])
            if similar:
                content.append("## Semantic Links")
                # Sort by similarity descending
                similar = sorted(similar, key=lambda x: x[1], reverse=True)
                for other_tag, sim in similar:
                    link_name = f"{field} - {other_tag}"
                    content.append(f"- [[{sanitize_filename(link_name)}]] (Similarity: {sim:.2f})")
                content.append("")
                
            filename = sanitize_filename(f"{field} - {tag}.md")
            with open(VAULT_DIR / "Entities" / filename, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
                
    # Generate Atoms
    print("Generating Atoms...")
    article_to_atoms = {}
    for _, atom in df.iterrows():
        atom_id = atom["atom_id"]
        article_id = atom["article_id"]
        
        if article_id not in article_to_atoms:
            article_to_atoms[article_id] = []
        article_to_atoms[article_id].append(atom_id)
        
        frontmatter = {
            "type": "atom",
            "article": f"[[{sanitize_filename('Article ' + article_id)}]]",
            "condition_type": atom["condition_type"],
            "lid": atom["lid"]
        }
        
        content = [
            "---",
            yaml.dump(frontmatter, sort_keys=False).strip(),
            "---",
            f"# Atom: {atom_id}",
            "",
            f"**Text**: {atom['text']}",
            f"**Text (NL)**: {atom['text_nl']}",
            "",
            "## Entities",
        ]
        
        for field in LIST_FIELDS:
            tags = atom.get(field, [])
            if tags:
                content.append(f"### {field.replace('_', ' ').title()}")
                for tag in tags:
                    link_name = f"{field} - {tag}"
                    content.append(f"- [[{sanitize_filename(link_name)}]]")
                content.append("")
                
        if atom.get("notes"):
            content.append("## Notes")
            content.append(atom["notes"])
            
        filename = sanitize_filename(f"Atom {atom_id}.md")
        with open(VAULT_DIR / "Atoms" / filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    # Generate Articles
    print("Generating Articles...")
    for article_id, atoms in article_to_atoms.items():
        frontmatter = {
            "type": "article",
            "article_id": article_id
        }
        content = [
            "---",
            yaml.dump(frontmatter, sort_keys=False).strip(),
            "---",
            f"# Article {article_id}",
            "",
            "## Atoms",
        ]
        for atom_id in sorted(atoms):
            content.append(f"- [[{sanitize_filename('Atom ' + atom_id)}]]")
            
        filename = sanitize_filename(f"Article {article_id}.md")
        with open(VAULT_DIR / "Articles" / filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
            
    print("Done! Obsidian vault generated successfully.")

if __name__ == "__main__":
    main()
