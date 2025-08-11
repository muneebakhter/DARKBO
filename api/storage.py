import json
import fcntl
import hashlib
import tempfile
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from filelock import FileLock
from datetime import datetime

from api.models import FAQEntry, KBEntry, ProjectMetadata, IndexMetadata


class FileStorageManager:
    """Manages file storage with atomic writes and locking"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.home()
        self.proj_mapping_file = self.base_dir / "proj_mapping.txt"
        
    def _ensure_project_dir(self, project_id: str) -> Path:
        """Ensure project directory exists"""
        project_dir = self.base_dir / project_id
        project_dir.mkdir(exist_ok=True)
        (project_dir / "attachments").mkdir(exist_ok=True)
        (project_dir / "index" / "dense").mkdir(parents=True, exist_ok=True)
        (project_dir / "index" / "sparse").mkdir(parents=True, exist_ok=True)
        return project_dir
    
    def _atomic_write_json(self, file_path: Path, data: List[Dict]) -> None:
        """Atomically write JSON data to file"""
        lock_file = str(file_path) + ".lock"
        
        with FileLock(lock_file, timeout=30):
            # Write to temporary file first
            temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
            
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename
                temp_file.replace(file_path)
                
            except Exception:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def _read_json_safe(self, file_path: Path) -> List[Dict]:
        """Safely read JSON file with locking"""
        if not file_path.exists():
            return []
        
        lock_file = str(file_path) + ".lock"
        
        with FileLock(lock_file, timeout=30):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except (json.JSONDecodeError, FileNotFoundError):
                return []
    
    def _calculate_checksum(self, data: List[Dict]) -> str:
        """Calculate MD5 checksum of JSON data"""
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def load_project_mapping(self) -> Dict[str, str]:
        """Load project mapping from file"""
        if not self.proj_mapping_file.exists():
            return {}
        
        projects = {}
        try:
            with open(self.proj_mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        project_id, project_name = line.split('\t', 1)
                        projects[project_id.strip()] = project_name.strip()
        except FileNotFoundError:
            pass
        
        return projects
    
    def save_project_mapping(self, projects: Dict[str, str]) -> None:
        """Save project mapping to file"""
        lock_file = str(self.proj_mapping_file) + ".lock"
        
        with FileLock(lock_file, timeout=30):
            temp_file = self.proj_mapping_file.with_suffix(".tmp")
            
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for project_id, project_name in projects.items():
                        f.write(f"{project_id}\t{project_name}\n")
                    f.flush()
                    os.fsync(f.fileno())
                
                temp_file.replace(self.proj_mapping_file)
                
            except Exception:
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def create_or_update_project(self, project_id: str, project_name: str) -> bool:
        """Create or update a project"""
        projects = self.load_project_mapping()
        is_new = project_id not in projects
        projects[project_id] = project_name
        self.save_project_mapping(projects)
        
        # Ensure project directory exists
        self._ensure_project_dir(project_id)
        
        return is_new
    
    def load_faqs(self, project_id: str) -> List[FAQEntry]:
        """Load FAQ entries for a project"""
        project_dir = self._ensure_project_dir(project_id)
        faq_file = project_dir / f"{project_id}.faq.json"
        
        data = self._read_json_safe(faq_file)
        return [FAQEntry(**item) for item in data]
    
    def save_faqs(self, project_id: str, faqs: List[FAQEntry]) -> None:
        """Save FAQ entries for a project"""
        project_dir = self._ensure_project_dir(project_id)
        faq_file = project_dir / f"{project_id}.faq.json"
        
        data = [faq.model_dump() for faq in faqs]
        self._atomic_write_json(faq_file, data)
        
        # Update index metadata
        self._update_index_metadata(project_id, faq_checksum=self._calculate_checksum(data))
    
    def load_kb_entries(self, project_id: str) -> List[KBEntry]:
        """Load KB entries for a project"""
        project_dir = self._ensure_project_dir(project_id)
        kb_file = project_dir / f"{project_id}.kb.json"
        
        data = self._read_json_safe(kb_file)
        return [KBEntry(**item) for item in data]
    
    def save_kb_entries(self, project_id: str, kb_entries: List[KBEntry]) -> None:
        """Save KB entries for a project"""
        project_dir = self._ensure_project_dir(project_id)
        kb_file = project_dir / f"{project_id}.kb.json"
        
        data = [entry.model_dump() for entry in kb_entries]
        self._atomic_write_json(kb_file, data)
        
        # Update index metadata
        self._update_index_metadata(project_id, kb_checksum=self._calculate_checksum(data))
    
    def upsert_faqs(self, project_id: str, new_faqs: List[FAQEntry], replace: bool = False) -> Tuple[List[str], List[str]]:
        """Upsert FAQ entries, returning (created_ids, updated_ids)"""
        if replace:
            existing_faqs = []
        else:
            existing_faqs = self.load_faqs(project_id)
        
        existing_by_id = {faq.id: faq for faq in existing_faqs}
        created_ids = []
        updated_ids = []
        
        for new_faq in new_faqs:
            new_faq.updated_at = datetime.utcnow()
            
            if new_faq.id in existing_by_id:
                updated_ids.append(new_faq.id)
            else:
                created_ids.append(new_faq.id)
            
            existing_by_id[new_faq.id] = new_faq
        
        final_faqs = list(existing_by_id.values())
        self.save_faqs(project_id, final_faqs)
        
        return created_ids, updated_ids
    
    def upsert_kb_entries(self, project_id: str, new_entries: List[KBEntry]) -> Tuple[List[str], List[str]]:
        """Upsert KB entries, returning (created_ids, updated_ids)"""
        existing_entries = self.load_kb_entries(project_id)
        existing_by_id = {entry.id: entry for entry in existing_entries}
        
        created_ids = []
        updated_ids = []
        
        for new_entry in new_entries:
            new_entry.updated_at = datetime.utcnow()
            
            if new_entry.id in existing_by_id:
                updated_ids.append(new_entry.id)
            else:
                created_ids.append(new_entry.id)
            
            existing_by_id[new_entry.id] = new_entry
        
        final_entries = list(existing_by_id.values())
        self.save_kb_entries(project_id, final_entries)
        
        return created_ids, updated_ids
    
    def _update_index_metadata(self, project_id: str, faq_checksum: str = None, kb_checksum: str = None) -> None:
        """Update index metadata"""
        project_dir = self._ensure_project_dir(project_id)
        meta_file = project_dir / "index" / "meta.json"
        
        # Load existing metadata
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta_dict = json.load(f)
                    metadata = IndexMetadata(**meta_dict)
            except (json.JSONDecodeError, FileNotFoundError):
                metadata = IndexMetadata()
        else:
            metadata = IndexMetadata()
        
        # Update checksums if provided
        if faq_checksum is not None:
            metadata.faq_checksum = faq_checksum
            metadata.faq_count = len(self.load_faqs(project_id))
        
        if kb_checksum is not None:
            metadata.kb_checksum = kb_checksum
            metadata.kb_count = len(self.load_kb_entries(project_id))
        
        metadata.last_updated = datetime.utcnow()
        
        # Save metadata
        with open(meta_file, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)
    
    def get_index_metadata(self, project_id: str) -> IndexMetadata:
        """Get index metadata for a project"""
        project_dir = self._ensure_project_dir(project_id)
        meta_file = project_dir / "index" / "meta.json"
        
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta_dict = json.load(f)
                    return IndexMetadata(**meta_dict)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return IndexMetadata()
    
    def save_attachment(self, project_id: str, filename: str, content: bytes) -> Path:
        """Save uploaded file to attachments directory"""
        project_dir = self._ensure_project_dir(project_id)
        attachments_dir = project_dir / "attachments"
        
        # Generate unique filename to avoid conflicts
        file_path = attachments_dir / filename
        counter = 1
        base_name = Path(filename).stem
        suffix = Path(filename).suffix
        
        while file_path.exists():
            new_name = f"{base_name}_{counter}{suffix}"
            file_path = attachments_dir / new_name
            counter += 1
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return file_path