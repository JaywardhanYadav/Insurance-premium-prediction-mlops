from pathlib import Path
import mlflow
from src.config_loader import load_config, get_project_root

cfg = load_config()
tracking_uri = cfg['paths']['mlflow_tracking_uri']
print('tracking_uri', tracking_uri)
if tracking_uri.startswith('sqlite:///') and not tracking_uri.startswith('sqlite:////'):
    db_file = tracking_uri.replace('sqlite:///', '')
    tracking_uri = f"sqlite:///{(get_project_root() / db_file).as_posix()}"
print('resolved', tracking_uri)
mlflow.set_tracking_uri(tracking_uri)
exp = mlflow.get_experiment_by_name(cfg['mlflow']['experiment_name'])
print('experiment_id', exp.experiment_id, 'name', exp.name)
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=['start_time DESC'], max_results=5)
print(runs[['run_id','status','start_time']].to_string(index=False))
