import os
import json
import sys
import time
import platform
from config.settings import Settings
from data.project_loader import ProjectLoader
from data.vector_store import VectorStoreManager

# Timeout configuration (in seconds)
PROJECT_TIMEOUT = 1800  # 30 minutes

class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs=None, timeout=30):
    """Run a function with a timeout in a cross-platform way."""
    kwargs = kwargs or {}
    if platform.system() == "Windows":
        import threading

        result = [None]
        exception = [None]
        finished = threading.Event()

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                finished.set()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        finished.wait(timeout)

        if not finished.is_set():
            raise TimeoutException(f"Operation timed out after {timeout} seconds")

        if exception[0]:
            raise exception[0]

        return result[0]
    else:
        import signal

        class UnixTimeoutException(Exception):
            pass

        def timeout_handler(signum, frame):
            raise UnixTimeoutException("Operation timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)
            return result
        except UnixTimeoutException:
            raise TimeoutException(f"Operation timed out after {timeout} seconds")
        finally:
            signal.alarm(0)

def build_all_vector_stores():
    """Prebuild vector stores for all projects with resume and timeout support."""
    settings = Settings()
    project_loader = ProjectLoader(settings)
    vector_store_manager = VectorStoreManager(settings, project_loader)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, 'vector_store_progress.json')

    skip_file = os.path.join(checkpoint_dir, 'timed_out_projects.txt')
    timed_out_projects = set()
    if os.path.exists(skip_file):
        try:
            with open(skip_file, 'r') as f:
                timed_out_projects = set(line.strip() for line in f if line.strip())
            print(f"\nLoaded {len(timed_out_projects)} projects from skip list")
        except Exception as e:
            print(f"Error loading skip list: {str(e)}")

    last_processed = None
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                last_processed = checkpoint_data.get('last_processed')
            print(f"\nCheckpoint found. Resuming from project: {last_processed}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting from the beginning...")

    all_projects = list(project_loader.projects.keys())
    total_projects = len(all_projects)

    if last_processed is not None and last_processed in all_projects:
        start_index = all_projects.index(last_processed) + 1
        projects_to_process = all_projects[start_index:]
        print(f"\nFound {start_index} completed projects. Processing remaining {len(projects_to_process)}/{total_projects} projects...")
    else:
        projects_to_process = all_projects
        print(f"\nNo valid checkpoint found. Processing all {total_projects} projects...")

    original_count = len(projects_to_process)
    projects_to_process = [p for p in projects_to_process if p not in timed_out_projects]
    skipped_count = original_count - len(projects_to_process)
    if skipped_count > 0:
        print(f"Skipped {skipped_count} projects that previously timed out")

    if not projects_to_process:
        print("\nAll vector stores are already built or skipped. Nothing to do.")
        return

    print(f"\nBuilding vector stores for {len(projects_to_process)} projects...")

    for i, project_id in enumerate(projects_to_process, 1):
        print(f"\n{'='*50}")
        print(f"Building vector store for project {project_id} ({i}/{len(projects_to_process)}): {project_loader.projects[project_id]}")
        print(f"{'='*50}")

        try:
            start_time = time.time()
            print(f"Starting vector store creation with {PROJECT_TIMEOUT} seconds timeout...")

            vector_store = run_with_timeout(
                vector_store_manager.create_vector_store,
                args=(project_id,),
                timeout=PROJECT_TIMEOUT
            )

            elapsed = time.time() - start_time
            print(f"Vector store creation completed in {elapsed:.2f} seconds")

            if vector_store:
                from rag.retriever import TwoStageRetriever
                retriever = TwoStageRetriever(vector_store, project_id, settings)
                results = retriever.test_retrieval("test query")
                print(f"Retrieved {len(results)} test documents")

                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_processed': project_id}, f)
                print(f"\u2713 Checkpoint updated. Project {project_id} completed.")
            else:
                print(f"Vector store creation failed for project {project_id}")

        except TimeoutException as e:
            print(f"\n!!! TIMEOUT: {str(e)}")
            timed_out_projects.add(project_id)
            with open(skip_file, 'a') as f:
                f.write(f"{project_id}\n")
            print(f"Added project {project_id} to skip list for future runs")
            continue
        except Exception as e:
            print(f"Error building vector store for project {project_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"\n!!! Processing stopped at project {project_id}")
            print(f"!!! Restart the script to resume from this point")
            sys.exit(1)

    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("\nAll vector stores built successfully. Checkpoint file removed.")
        except Exception as e:
            print(f"\nWarning: Could not remove checkpoint file: {str(e)}")

    print("\nVector store prebuild complete!")

if __name__ == "__main__":
    build_all_vector_stores()
