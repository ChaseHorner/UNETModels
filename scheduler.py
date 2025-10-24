import subprocess
import time
import os
import re
import json

model_names = ['UNET_v1.1.1', 'UNET_v1.2.1', 'UNET_v1.3.1', 'UNET_v1.4.1']



def submit_training_job(model_name):
    """
    Submits a SLURM job for a given model.
    """

    config_path = f"/kuhpc/scratch/kbs/c710h797/UNETModels/outputs/{model_name}/configs.py"

    # The bash command to submit the job using `sbatch`
    command = ['sbatch', f'--export=CONFIG={config_path}', f'--output=outputs/{model_name}/slurm-%j.out',
               f'--error=outputs/{model_name}/slurm-%j.err', f'--job-name="{model_name}"', 'run.sh', model_name]
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
    model_statuses = {}
    
    while True:
        models_to_resubmit = []
        
        # Check all models for their finished status
        for model_name in model_names:
            model_status_path = f'outputs/{model_name}/status.json'
            with open(model_status_path, "r") as f:
                model_status = json.load(f)
                model_statuses[model_name] = model_status
            if model_status["finished"]:
                print(f"Model {model_name} is already finished.")
                if model_name in current_jobs:
                    del current_jobs[model_name]
            elif model_name not in current_jobs:
                models_to_resubmit.append(model_name)
        
        # Check active jobs for any that have ended without finishing
        jobs_to_delete = []
        for model_name, job_id in current_jobs.items():
            try:
                sacct_command = ['sacct', '-j', str(job_id), '--format=State', '--noheader']
                sacct_result = subprocess.run(sacct_command, capture_output=True, text=True, check=True)
                status = sacct_result.stdout.strip().split('\n')
                
                print(f"Job {job_id} for {model_name} is in state: {status}")

                if 'COMPLETED' in status or 'TIMEOUT' in status or 'FAILED' in status or 'CANCELLED' in status:
                    # The job is no longer running.
                    # Since it wasn't marked as finished, we must resubmit it.
                    print(f"Job {job_id} for {model_name} ended. Adding to resubmit list.")
                    models_to_resubmit.append(model_name)
                    jobs_to_delete.append(model_name)
            
            except subprocess.CalledProcessError:
                # `sacct` will fail for jobs that are too old and have been purged from the database.
                # In this case, we'll assume it finished and wasn't marked complete.
                print(f"Could not retrieve status for job {job_id}. Assuming ended and needs resubmission.")
                models_to_resubmit.append(model_name)
                jobs_to_delete.append(model_name)

        for model in jobs_to_delete:
            del current_jobs[model]

        # Submit new jobs for any models that need it
        for model_name in models_to_resubmit:
            job_id = submit_training_job(model_name)
            if job_id:
                current_jobs[model_name] = job_id

        # Exit condition: all models are finished
        if not current_jobs and all(m in model_statuses and model_statuses[m]["finished"] for m in model_names):
            print("All models have finished training.")
            break
        
        print(f"Currently monitoring {len(current_jobs)} active jobs.")
        time.sleep(60 * 20)  # Wait for 20 minutes before checking again

if __name__ == "__main__":
    for model_name in model_names:
        model_status = {
        "model_name": model_name,
        "finished": False,
        "metrics": None,
        "model_path": None,
        "optimizer_path": None,
        "early_stopping": False,
        "last_trained_epoch": 0,
        "last_trained_time": None
    }
        model_status_path = f'outputs/{model_name}/status.json'
        with open(model_status_path, "w") as f:
            json.dump(model_status, f)


    monitor_jobs()
