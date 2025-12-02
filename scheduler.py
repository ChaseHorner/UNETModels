import subprocess
import time
import os
import re
import json


# model_paths = ['instancenorm/u4ai_24', 'instancenorm/u4ai_24_auc', 'instancenorm/u16ai_24', 'instancenorm/u1.2.1i_24']
model_paths = [
    'thanksgiving/u4_SHT16',
    'thanksgiving/u4ASHT8',
    'thanksgiving/u4_SLT8',
    'thanksgiving/u4ASLT8',
    
    'jump_models/u4JA',
    'jump_models/u4JB',
    'jump_models/uS4JA',
    'jump_models/uS4JB',
    'jump_models/uS16JA',
    'jump_models/uS16JB']


def submit_training_job(model_path):
    """
    Submits a SLURM job for a given model.
    """

    model_name = model_path.split('/')[-1]
    config_path = f"/kuhpc/work/kbs/c710h797/UNETModels/outputs/{model_path}/configs.py"

    # The bash command to submit the job using `sbatch`
    command = ['sbatch', f'--export=CONFIG={config_path}', f'--output=outputs/{model_path}/slurm-%j.out',
               f'--error=outputs/{model_path}/slurm-%j.err', f'--job-name="{model_name}"', 'run.sh', model_path]
    print(f"Submitting job for {model_name}...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id_match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Job submitted for {model_name} with ID: {job_id}")
            return job_id
        else:
            print("Could not find job ID in sbatch output.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job for {model_name}: {e.stderr}")
        return None

def monitor_jobs():
    """
    Monitors all models and resubmits jobs for unfinished ones.
    """
    current_jobs = {}
    
    while True:
        models_to_submit_or_resubmit = []
        
        # Check all models for their finished status based on status.json
        for model_path in model_paths:
            model_name = model_path.split('/')[-1]
            model_status_path = f'outputs/{model_path}/status.json'
            try:
                with open(model_status_path, "r") as f:
                    model_status = json.load(f)
            except FileNotFoundError:
                print(f"Status file not found for {model_name}. Skipping...")
                continue
            
            if model_status["finished"]:
                print(f"Model {model_name} is already finished.")
                if model_path in current_jobs:
                    del current_jobs[model_path]
            elif model_path not in current_jobs:
                models_to_submit_or_resubmit.append(model_path)
        
        # Check active jobs for any that have ended without finishing
        # Iterate over a copy of the dictionary to avoid issues with deleting items during iteration
        jobs_to_check = list(current_jobs.items())
        for model_path, job_id in jobs_to_check:
            try:
                # Use --noheader and -n for a cleaner state output
                sacct_command = ['sacct', '-j', str(job_id), '--format=State', '--noheader', '-n']
                sacct_result = subprocess.run(sacct_command, capture_output=True, text=True, check=True)
                
                # Check for any state that indicates the job is no longer active
                # The output from sacct can include step information, so check all lines
                states = [line.strip() for line in sacct_result.stdout.strip().split('\n') if line.strip()]
                
                print(f"Job {job_id} for {model_path} is in state: {states}")

                # If the job is not PENDING or RUNNING, it has ended.
                if not any(s in states for s in ['PENDING', 'RUNNING', 'FAILED', 'CANCELLED']):
                    print(f"Job {job_id} for {model_path} ended. Adding to resubmit list.")
                    if model_path not in models_to_submit_or_resubmit:
                        models_to_submit_or_resubmit.append(model_path)
                    del current_jobs[model_path]

                # If the job has failed or cancelled, mark it as finished
                if any(s in states for s in ['FAILED']):
                    print(f"Job {job_id} for {model_path} has failed or was cancelled. Marking as finished.")
                    with open(f'outputs/{model_path}/status.json', "r") as f:
                        model_status = json.load(f)
                    model_status["errored"] = True
                    with open(f'outputs/{model_path}/status.json', "w") as f:
                        json.dump(model_status, f)
                    del current_jobs[model_path]
            
            except subprocess.CalledProcessError:
                # This could mean the job ID is no longer in Slurm's active accounting.
                # It likely ended long ago without updating the status file.
                print(f"Could not retrieve status for job {job_id}. Assuming ended and needs resubmission.")
                if model_path not in models_to_submit_or_resubmit:
                    models_to_submit_or_resubmit.append(model_path)
                del current_jobs[model_path]

        # Submit new jobs for any models that need it
        for model_path in models_to_submit_or_resubmit:
            job_id = submit_training_job(model_path)
            if job_id:
                current_jobs[model_path] = job_id

        # Exit condition: all models are finished
        all_finished = True
        for model_path in model_paths:
            try:
                with open(f'outputs/{model_path}/status.json', "r") as f:
                    model_status = json.load(f)
                if not model_status["finished"] or model_status["errored"]:
                    all_finished = False
                    break
            except FileNotFoundError:
                all_finished = False
                break
        
        if not current_jobs and all_finished:
            print("All models have finished training.")
            break
        
        print(f"Currently monitoring {len(current_jobs)} active jobs.")
        time.sleep(60 * 5)  # Wait a short time before checking again, can be increased.


if __name__ == "__main__":
    for model_path in model_paths:
        model_name = model_path.split('/')[-1]

        model_status = {
        "model_name": model_name,
        "finished": False,
        "early_stopping": False,
        "errored": False,
        "metrics": {},
        "model_path": "",
        "optimizer_path": "",
        "last_trained_epoch": 0,
        "last_trained_time": "",
    }
        model_status_path = f'outputs/{model_path}/status.json'

        # If status file doesn't exist, create it
        if not os.path.exists(model_status_path):
            os.makedirs(os.path.dirname(model_status_path), exist_ok=True)
            with open(model_status_path, "w") as f:
                json.dump(model_status, f)

        # mark errored as false for rerun
        else:
            try:
                with open(model_status_path, "r") as f:
                    model_status = json.load(f)
            except (json.JSONDecodeError, OSError, ValueError):
                # File exists but is empty or corrupted → rebuild it
                model_status = {
                    "model_name": model_name,
                    "finished": False,
                    "early_stopping": False,
                    "errored": False,
                    "metrics": {},
                    "model_path": "",
                    "optimizer_path": "",
                    "last_trained_epoch": 0,
                    "last_trained_time": "",
                }

            # Update fields after load or rebuild
            model_status["errored"] = False

            with open(model_status_path, "w") as f:
                json.dump(model_status, f)


    monitor_jobs()
