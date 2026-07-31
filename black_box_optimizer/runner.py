"""
runner.py

Phase 2 of runtimeflow
This program will run the worker program and then give the data to metrics.py.
No parsing is done yet, this just executes the program.
It will then feed the following information for metrics parsing. 
"""

import subprocess
import time

def execute(worker_spec, candidate, metrics_path) -> dict[str, object]:
    """USAGE: result = execute(config.worker, candidate, metrics_path)"""
    #build the command
    command = list(worker_spec.command)
    #loop through all parameters, adding them to the command
    for name, value in candidate.parameters.items():
        flag = "--" + name.replace("_", "-")
        command.extend([flag, str(value)])
        
    command.extend([
        worker_spec.metrics_argument,
        str(metrics_path)
    ])

    start_time = time.perf_counter()
    #Failure Handling
    #these are conditions for a completed run
    exit_code = None
    timed_out = False
    execution_status = "completed"
    error_message = None
    try:
        #create subprocess
        process = subprocess.run(
            command,
            shell = False,
            timeout = worker_spec.timeout_seconds,
            capture_output = True,
            text = True
        )
        exit_code = process.returncode
        if exit_code != 0:
            execution_status = "process_failed"
        
    except FileNotFoundError as error:
        execution_status = "launch_failed"
        error_message = str(error)
    except subprocess.TimeoutExpired as error:
        execution_status = "timed_out"
        timed_out = True
        error_message = str(error)

    #return observations
    runtime = time.perf_counter() - start_time
    return {
        "runtime_seconds" : runtime,
        "exit_code" : exit_code,
        "timed_out" : timed_out,
        "execution_status" : execution_status,
        "error_message" : error_message
    }