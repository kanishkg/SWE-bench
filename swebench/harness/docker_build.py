from __future__ import annotations

import logging
import re
import sys
import subprocess
import traceback
from pathlib import Path

from swebench.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    DOCKER_USER,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    UTF8,
)
from swebench.harness.docker_utils import (
    cleanup_container,
    remove_image,
    find_dependent_images
)
from swebench.harness.test_spec.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec,
)
from swebench.harness.utils import run_threadpool

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )

def setup_logger(instance_id: str, log_file: Path, mode="w", add_stdout: bool = False):
    """Logger setup remains the same as original"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    if add_stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(f"%(asctime)s - {instance_id} - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def close_logger(logger):
    """Remains the same as original"""
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def build_image(
        image_name: str,
        setup_scripts: dict,
        dockerfile: str,  # Keep dockerfile parameter name but use as definition file
        platform: str,
        client: None,  # Keep for compatibility but don't use
        build_dir: Path,
        nocache: bool = False
    ):
    """
    Builds a Singularity image instead of Docker image.
    """
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building Singularity image {image_name}\n"
        f"Using definition file:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts"
    )

    try:
        # Write setup scripts
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)

        # Write Singularity definition file (converted from Dockerfile)
        def_file = convert_dockerfile_to_definition(dockerfile)
        def_file_path = build_dir / "Singularity.def"
        with open(def_file_path, "w") as f:
            f.write(def_file)

        # Build image
        sif_path = build_dir / f"{image_name.replace(':', '_')}.sif"
        cmd = ["singularity", "build"]
        if nocache:
            cmd.append("--force")
        cmd.extend(["--fakeroot", str(sif_path), str(def_file_path)])
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        buildlog = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                clean_output = ansi_escape.sub("", output.strip())
                logger.info(clean_output)
                buildlog += clean_output

        if process.returncode != 0:
            error = process.stderr.read()
            raise Exception(f"Build failed: {error}")

        logger.info("Image built successfully!")
        return sif_path

    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)

def build_base_images(
        client: None,  # Keep for compatibility but don't use
        dataset: list,
        force_rebuild: bool = False
    ):
    """
    Builds base Singularity images.
    """
    test_specs = get_test_specs_from_dataset(dataset)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }

    for image_name, (dockerfile, platform) in base_images.items():
        sif_path = BASE_IMAGE_BUILD_DIR / f"{image_name.replace(':', '_')}.sif"
        
        if sif_path.exists() and not force_rebuild:
            print(f"Base image {sif_path} already exists, skipping build.")
            continue

        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=None,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")

def build_env_images(
        client: None,  # Keep for compatibility but don't use
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4
    ):
    """
    Builds environment Singularity images.
    """
    if force_rebuild:
        env_image_keys = {x.env_image_key for x in get_test_specs_from_dataset(dataset)}
        for key in env_image_keys:
            sif_path = ENV_IMAGE_BUILD_DIR / f"{key.replace(':', '_')}.sif"
            if sif_path.exists():
                sif_path.unlink()

    build_base_images(None, dataset, force_rebuild)
    configs_to_build = get_env_configs_to_build(None, dataset)
    
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return [], []

    print(f"Total environment images to build: {len(configs_to_build)}")

    args_list = list()
    for image_name, config in configs_to_build.items():
        args_list.append((
            image_name,
            {"setup_env.sh": config["setup_script"]},
            config["dockerfile"],
            config["platform"],
            None,
            ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        ))
    
    successful, failed = run_threadpool(build_image, args_list, max_workers)
    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    return successful, failed

def get_env_configs_to_build(
        client: None,  # Keep for compatibility but don't use
        dataset: list,
    ):
    """
    Gets configurations for environment images that need building.
    """
    image_scripts = dict()
    test_specs = get_test_specs_from_dataset(dataset)

    for test_spec in test_specs:
        # Check if base image exists
        base_sif = BASE_IMAGE_BUILD_DIR / f"{test_spec.base_image_key.replace(':', '_')}.sif"
        if not base_sif.exists():
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n"
                "Please build the base images first."
            )

        # Check if env image exists
        env_sif = ENV_IMAGE_BUILD_DIR / f"{test_spec.env_image_key.replace(':', '_')}.sif"
        if not env_sif.exists():
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts

def build_instance_images(
        client: None,  # Keep for compatibility but don't use
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4,
        namespace: str = None,
        tag: str = None,
    ):
    """
    Builds instance Singularity images.
    """
    test_specs = list(map(lambda x: make_test_spec(x, namespace=namespace, instance_image_tag=tag), dataset))
    if force_rebuild:
        for spec in test_specs:
            sif_path = INSTANCE_IMAGE_BUILD_DIR / f"{spec.instance_image_key.replace(':', '_')}.sif"
            if sif_path.exists():
                sif_path.unlink()

    _, env_failed = build_env_images(None, test_specs, force_rebuild, max_workers)

    if len(env_failed) > 0:
        dont_run_specs = [spec for spec in test_specs if spec.env_image_key in env_failed]
        test_specs = [spec for spec in test_specs if spec.env_image_key not in env_failed]
        print(f"Skipping {len(dont_run_specs)} instances - due to failed env image builds")

    print(f"Building instance images for {len(test_specs)} instances")
    
    payloads = [(spec, None, False) for spec in test_specs]
    successful, failed = run_threadpool(build_instance_image, payloads, max_workers)

    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    return successful, failed

def build_instance_image(
        test_spec: TestSpec,
        client: None,  # Keep for compatibility but don't use
        logger: logging.Logger|None,
        nocache: bool,
    ):
    """
    Builds instance Singularity image.
    """
    print(f"Building instance image for {test_spec.instance_id}...")
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"Build directory: {build_dir}")

    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key

    # Check that env image exists
    env_sif = ENV_IMAGE_BUILD_DIR / f"{env_image_name.replace(':', '_')}.sif"
    if not env_sif.exists():
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment image {env_image_name} not found for {test_spec.instance_id}",
            logger,
        )

    logger.info(
        f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name}"
    )

    # Check if instance image exists
    instance_sif = build_dir / f"{image_name.replace(':', '_')}.sif"
    if not instance_sif.exists():
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": test_spec.install_repo_script,
            },
            dockerfile=test_spec.instance_dockerfile,
            platform=test_spec.platform,
            client=None,
            build_dir=build_dir,
            nocache=nocache,
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)

def build_container(
        test_spec: TestSpec,
        client: None,  # Keep for compatibility but don't use
        run_id: str,
        logger: logging.Logger,
        nocache: bool,
        force_rebuild: bool = False
    ):
    """
    Creates Singularity instance instead of Docker container.
    """
    if force_rebuild:
        sif_path = INSTANCE_IMAGE_BUILD_DIR / f"{test_spec.instance_image_key.replace(':', '_')}.sif"
        if sif_path.exists():
            sif_path.unlink()

    if not test_spec.is_remote_image:
        print(f"Building instance image for {test_spec.instance_id}...")
        build_instance_image(test_spec, None, logger, nocache)
    else:
        # For remote images, pull using singularity
        print(f"Pulling image {test_spec.instance_image_key}...")
        try:
            cmd = ["singularity", "pull", f"{test_spec.instance_image_key}.sif", f"docker://{test_spec.instance_image_key}"]
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise Exception(f"Failed to pull image: {process.stderr}")
        except Exception as e:
            raise BuildImageError(test_spec.instance_id, str(e), logger) from e

    try:
        logger.info(f"Creating Singularity instance for {test_spec.instance_id}...")
        
        instance_name = test_spec.get_instance_container_name(run_id)
        sif_path = INSTANCE_IMAGE_BUILD_DIR / f"{test_spec.instance_image_key.replace(':', '_')}.sif"
        
        cmd = [
            "singularity",
            "instance",
            "start",
            str(sif_path),
            instance_name
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise Exception(f"Failed to start instance: {process.stderr}")

        logger.info(f"Instance {instance_name} created successfully")
        return instance_name

    except Exception as e:
        logger.error(f"Error creating instance for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        # Try to stop instance if it exists
        try:
            subprocess.run(["singularity", "instance", "stop", instance_name], capture_output=True)
        except:
            pass
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e

def parse_docker_command(line: str) -> tuple[str, str]:
    """Helper function to parse Docker commands that might contain spaces in values"""
    line = line.strip()
    parts = line.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

def convert_dockerfile_to_definition(dockerfile: str) -> str:
    """
    Converts a Dockerfile to a Singularity definition file using spython.
    
    Args:
        dockerfile (str): Original Dockerfile content
        
    Returns:
        str: Singularity definition file content
    """
    # Create temporary dockerfile to parse
    dockerfile_path = Path("temp_dockerfile")
    try:
        dockerfile_path.write_text(dockerfile)
        
        # Parse dockerfile using spython
        recipe = DockerRecipe(dockerfile_path)
        singularity_def = recipe.convert()
        
        # Add startscript section which wasn't in original spython conversion
        if not '%startscript' in singularity_def:
            singularity_def += '\n\n%startscript\n    exec tail -f /dev/null'
            
        return singularity_def
        
    finally:
        if dockerfile_path.exists():
            dockerfile_path.unlink()