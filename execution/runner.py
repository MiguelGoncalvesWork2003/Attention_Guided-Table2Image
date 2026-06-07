#runner.py
"""
Generic subprocess‑based pipeline runner with error handling and logging.

This module encapsulates all subprocess execution logic used to orchestrate the
distinct stages of the attention‑guided tabular‑to‑image framework. It provides
a uniform, reproducible interface for invoking preprocessing, TabNet training,
layout construction, CNN training, and visualisation steps as separate,
isolated processes.

Key features:
  - `run_step`: Executes a single Python script as a subprocess, capturing
    stdout and stderr, enforcing timeouts, and optionally raising a custom
    `PipelineStepError` on failure.
  - `run_multiple_steps`: Sequentially runs a list of steps, with the ability
    to stop on first error.
  - `prepare_environment`: Builds a consistent `PYTHONPATH` and encoding
    environment, ensuring that subprocesses import project modules correctly.
  - `clean_output`: Normalises special Unicode characters for safe display in
    terminal and Streamlit logs.

**Role in the Map–Optimize–Learn pipeline:**
  - Serves as the execution backbone of the interactive Streamlit application
    and of any head‑less experiment scripts.
  - Guarantees that the **exact same Python code** is invoked regardless of
    whether the pipeline is run interactively or from the command line – a
    crucial point for the paper’s emphasis on reproducibility.
  - The subprocess isolation ensures that no global state leaks between stages,
    preventing, for example, accidental feedback from the CNN stage into the
    TabNet optimisation.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import traceback

class PipelineStepError(Exception):
    """Custom exception for pipeline step failures."""
    pass

def prepare_environment() -> Dict[str, str]:
    """
    Prepare environment variables for subprocess execution.
    
    Returns:
        Dictionary of environment variables
    """
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    existing = env.get("PYTHONPATH", "")
    project_path = str(Path(__file__).parent.parent)

    env["PYTHONPATH"] = (
        f"{project_path}{os.pathsep}{existing}"
        if existing else project_path
    )
    return env

def clean_output(output: str) -> str:
    """
    Clean and normalize subprocess output for display.
    
    Args:
        output: Raw output from subprocess
        
    Returns:
        Cleaned output with special characters replaced
    """
    # Replace special characters for clean display
    replacements = {
        '\u2714': '[OK]',
        '\u2718': '[FAIL]',
        '\u2192': '->',
        '\u25b6': '▶',
        '\u23f3': '⏳',
        '\u2705': '[SUCCESS]',
        '\u274c': '[ERROR]'
    }
    
    for char, replacement in replacements.items():
        output = output.replace(char, replacement)
    
    return output

def run_step(
    name: str,
    script_path: Path,
    env_vars: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    check_returncode: bool = True
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Execute a pipeline step as a subprocess.
    
    Args:
        name: Human-readable name of the step
        script_path: Path to the Python script to run
        env_vars: Additional environment variables to set
        timeout: Maximum execution time in seconds
        check_returncode: Whether to raise exception on non-zero return code
        
    Returns:
        Tuple of (success, output_message, metadata)
        
    Raises:
        PipelineStepError: If step fails and check_returncode is True
    """
    # Validate script exists
    if not script_path.exists():
        error_msg = f"Script not found: {script_path}"
        return False, error_msg, None
    
    # Prepare environment
    env = prepare_environment()
    if env_vars:
        env.update(env_vars)
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=check_returncode,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        
        # Prepare metadata
        metadata = {
            'returncode': result.returncode,
            'script': str(script_path),
            'success': result.returncode == 0
        }
        
        # Clean and format output
        if result.returncode == 0:
            output = f"✓ {name} completed successfully\n\n"
        else:
            output = f"✗ {name} failed with return code {result.returncode}\n\n"
        
        if result.stdout:
            output += clean_output(result.stdout)
        
        if result.stderr:
            output += "\n\n=== STDERR ===\n"
            output += clean_output(result.stderr)
        
        return result.returncode == 0, output, metadata
        
    except subprocess.CalledProcessError as e:
        error_msg = f"✗ {name} failed with return code {e.returncode}\n"
        if e.stderr:
            error_msg += f"\nError:\n{clean_output(e.stderr[-2000:])}"
        elif e.stdout:
            error_msg += f"\nOutput:\n{clean_output(e.stdout[-2000:])}"
        
        if check_returncode:
            raise PipelineStepError(error_msg) from e
        return False, error_msg, {'returncode': e.returncode}
        
    except subprocess.TimeoutExpired:
        error_msg = f"✗ {name} timed out after {timeout} seconds"
        if check_returncode:
            raise PipelineStepError(error_msg)
        return False, error_msg, {'timeout': True}
        
    except Exception as e:
        error_msg = f"✗ {name} failed with unexpected error: {str(e)}\n"
        error_msg += f"\nTraceback:\n{traceback.format_exc()[-3000:]}"
        if check_returncode:
            raise PipelineStepError(error_msg) from e
        return False, error_msg, None

def run_multiple_steps(
    steps: List[Dict],
    stop_on_error: bool = True
) -> List[Tuple[bool, str, Optional[Dict]]]:
    """
    Execute multiple pipeline steps sequentially.
    
    Args:
        steps: List of step configurations, each with 'name' and 'script_path'
        stop_on_error: Whether to stop execution if a step fails
        
    Returns:
        List of results for each step
    """
    results = []
    
    for step_config in steps:
        success, output, metadata = run_step(
            name=step_config.get('name', 'Unnamed Step'),
            script_path=step_config['script_path'],
            env_vars=step_config.get('env_vars'),
            timeout=step_config.get('timeout', 300),
            check_returncode=stop_on_error
        )
        
        results.append((success, output, metadata))
        
        if not success and stop_on_error:
            break
    
    return results