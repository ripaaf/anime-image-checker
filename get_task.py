from repo_map import task_map
from model_classify import Classification


def get_task(task_name):
    if task_name not in task_map:
        raise ValueError(f"Task {task_name} is not recognized.")
    
    task_info = task_map[task_name]
    return Classification(*task_info)
