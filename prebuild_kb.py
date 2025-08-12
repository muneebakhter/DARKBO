#!/usr/bin/env python3
"""
Simplified prebuild script for knowledge graphs and vector stores.
Creates indexes per project folder based on FAQ and KB data with versioning support.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

# Import the new versioned IndexBuilder
from api.index_versioning import IndexBuilder


def load_project_mapping() -> Dict[str, str]:
    """Load project mapping from proj_mapping.txt"""
    projects = {}
    mapping_file = Path("proj_mapping.txt")
    
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    project_id, name = line.split('\t', 1)
                    projects[project_id.strip()] = name.strip()
    
    return projects


def main():
    """Main prebuild function"""
    print("🚀 DARKBO Knowledge Base Prebuild")
    print("=" * 50)
    
    # Load project mapping
    projects = load_project_mapping()
    
    if not projects:
        print("❌ No projects found in proj_mapping.txt")
        return
    
    print(f"📋 Found {len(projects)} projects to process")
    
    # Build indexes for each project using versioning system
    results = {}
    
    for project_id, project_name in projects.items():
        try:
            # Use new IndexBuilder with versioning
            builder = IndexBuilder(project_id, ".")
            new_version = builder.build_new_version()
            
            if new_version:
                results[project_id] = {"version": new_version, "success": True}
                print(f"✅ {project_id} ({project_name}): version {new_version}")
            else:
                print(f"ℹ️  {project_id} ({project_name}): indexes up to date")
                results[project_id] = {"success": True, "up_to_date": True}
                
        except Exception as e:
            print(f"❌ {project_id} ({project_name}): {e}")
            results[project_id] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Build Summary")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for project_id, result in results.items():
        project_name = projects.get(project_id, "Unknown")
        if "error" in result:
            print(f"❌ {project_id} ({project_name}): {result['error']}")
            failed += 1
        else:
            if result.get("up_to_date"):
                print(f"✅ {project_id} ({project_name}): up to date")
            else:
                version = result.get("version", "unknown")
                print(f"✅ {project_id} ({project_name}): version {version}")
            successful += 1
    
    print(f"\n🎉 Completed: {successful} successful, {failed} failed")
    
    try:
        # Check for dependencies
        import sentence_transformers
        import faiss
        import whoosh
        print("\n✅ All indexing dependencies available")
    except ImportError:
        print("\n💡 To enable full indexing, install dependencies:")
        print("   pip install sentence-transformers faiss-cpu whoosh")


if __name__ == "__main__":
    main()