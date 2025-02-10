from __future__ import annotations

import os
import signal
import tarfile
import threading
import time
import traceback
from pathlib import Path
import subprocess
import spython.main
from spython.main.execute import execute

from spython.instance import Instance

HEREDOC_DELIMITER = "EOF_1399519320"  # different from dataset HEREDOC_DELIMITERs!

def copy_to_container(instance: Instance, src: Path, dst: Path):
    """
    Copy a file from local to a Singularity container instance

    Args:
        instance (Instance): Singularity container instance to copy to
        src (Path): Source file path
        dst (Path): Destination file path in the container
    """
    # Check if destination path is valid
    if os.path.dirname(dst) == "":
        raise ValueError(
            f"Destination path parent directory cannot be empty!, dst: {dst}"
        )

    # Make directory if necessary
    execute(instance, ['mkdir', '-p', str(dst.parent)])
    
    # Copy file directly using Singularity's copy command
    subprocess.run(['singularity', 'copy', str(src), f"{instance.name}:{dst}"], check=True)

def write_to_container(instance: Instance, data: str, dst: Path):
    """
    Write a string to a file in a Singularity container
    """
    # Use echo with heredoc via shell execution
    command = f"cat <<'{HEREDOC_DELIMITER}' > {dst}\n{data}\n{HEREDOC_DELIMITER}"
    spython.main.execute(instance, ['sh', '-c', command])

def remove_image(client, image_path: str, logger=None):
    """
    Remove a Singularity image file.

    Args:
        client: Not used (kept for API compatibility)
        image_path (str): Path to the Singularity image file
        logger (logging.Logger): Logger to use for output. If None, print to stdout
    """
    if not logger:
        log_info = print
        log_error = print
        raise_error = True
    elif logger == "quiet":
        log_info = lambda x: None
        log_error = lambda x: None
        raise_error = True
    else:
        log_error = logger.info
        log_info = logger.info
        raise_error = False

    try:
        log_info(f"Attempting to remove image {image_path}...")
        os.remove(image_path)
        log_info(f"Image {image_path} removed.")
    except FileNotFoundError:
        log_info(f"Image {image_path} not found, removing has no effect.")
    except Exception as e:
        if raise_error:
            raise e
        log_error(
            f"Failed to remove image {image_path}: {e}\n" f"{traceback.format_exc()}"
        )

def cleanup_container(client, instance: Instance, logger):
    """
    Stop and remove a Singularity container instance.
    
    Args:
        client: Not used (kept for API compatibility)
        instance (Instance): Singularity instance to remove
        logger (logging.Logger): Logger to use for output. If None, print to stdout
    """
    if not instance:
        return

    if not logger:
        log_error = print
        log_info = print
        raise_error = True
    elif logger == "quiet":
        log_info = lambda x: None
        log_error = lambda x: None
        raise_error = True
    else:
        log_error = logger.info
        log_info = logger.info
        raise_error = False

    try:
        log_info(f"Attempting to stop instance {instance.name}...")
        instance.stop()
        log_info(f"Instance {instance.name} stopped.")
    except Exception as e:
        log_error(
            f"Failed to stop instance {instance.name}: {e}. Trying to forcefully kill..."
        )
        try:
            # Get the PID of the instance
            pid = instance.pid
            if pid > 0:
                log_info(
                    f"Forcefully killing instance {instance.name} with PID {pid}..."
                )
                os.kill(pid, signal.SIGKILL)
            else:
                log_error(f"PID for instance {instance.name}: {pid} - not killing.")
        except Exception as e2:
            if raise_error:
                raise e2
            log_error(
                f"Failed to forcefully kill instance {instance.name}: {e2}\n"
                f"{traceback.format_exc()}"
            )

def exec_run_with_timeout(instance: Instance, cmd: str, timeout: int|None=60):
    """
    Run a command in a Singularity instance with a timeout.

    Args:
        instance (Instance): Instance to run the command in
        cmd (str): Command to run
        timeout (int): Timeout in seconds
    """
    exec_result = b''
    exec_pid = None
    exception = None
    timed_out = False

    def run_command():
        nonlocal exec_result, exec_pid, exception
        try:
            process = spython.main.execute(
                instance,
                cmd.split(),
                stream=True,
                return_result=True
            )
            exec_pid = process.pid
            for line in process.stdout:
                exec_result += line.encode()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if exception:
        raise exception

    if thread.is_alive():
        if exec_pid is not None:
            spython.main.execute(instance, ['kill', '-TERM', str(exec_pid)], stream=False)
        timed_out = True
    
    end_time = time.time()
    return exec_result.decode(), timed_out, end_time - start_time

def find_dependent_images(client, image_path: str):
    """
    Find all Singularity images that are built upon the base image.
    Note: Singularity doesn't track image dependencies like Docker. This is a best-effort implementation.

    Args:
        client: Not used (kept for API compatibility)
        image_path (str): Path to the base image file
    """
    # In Singularity, we need to implement this differently as it doesn't track dependencies
    # This is a placeholder that could be implemented by parsing definition files
    return []

def list_images(client):
    """
    List all Singularity image files in the current directory and common Singularity locations.
    
    Args:
        client: Not used (kept for API compatibility)
    """
    image_paths = set()
    
    # Common Singularity image extensions
    extensions = {'.sif', '.simg'}
    
    # Search in common locations
    search_paths = [
        os.getcwd(),
        os.path.expanduser('~/.singularity'),
        '/usr/local/singularity'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        image_paths.add(os.path.join(root, file))
    
    return image_paths

def clean_images(
        client,
        prior_images: set,
        cache_level: str,
        clean: bool
    ):
    """
    Clean Singularity images based on cache level and clean flag.

    Args:
        client: Not used (kept for API compatibility)
        prior_images (set): Set of image paths that existed before the current run
        cache_level (str): Cache level to use
        clean (bool): Whether to clean images that existed before this run
    """
    images = list_images(client)
    removed = 0
    print("Cleaning cached images...")
    for image_path in images:
        if should_remove(image_path, cache_level, clean, prior_images):
            try:
                remove_image(client, image_path, "quiet")
                removed += 1
            except Exception as e:
                print(f"Error removing image {image_path}: {e}")
                continue
    print(f"Removed {removed} images.")

def should_remove(
        image_path: str,
        cache_level: str,
        clean: bool,
        prior_images: set
    ):
    """
    Determine if an image should be removed based on cache level and clean flag.
    """
    existed_before = image_path in prior_images
    image_name = os.path.basename(image_path)
    
    if image_name.startswith("sweb.base"):
        if cache_level in {"none"} and (clean or not existed_before):
            return True
    elif image_name.startswith("sweb.env"):
        if cache_level in {"none", "base"} and (clean or not existed_before):
            return True
    elif image_name.startswith("sweb.eval"):
        if cache_level in {"none", "base", "env"} and (clean or not existed_before):
            return True
    return False