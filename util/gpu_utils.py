import torch

def get_best_devices(model, max_gpus=1, min_vram=2.0):
    """
    Returns a list containing the selected GPU device IDs.

    The function checks all available GPUs, filters out those with less than 
    `min_vram` gigabytes of free VRAM, sorts the remaining GPUs by available VRAM (descending),
    and returns the top `max_gpus` device IDs.

    Args:
        model: The model being trained (currently unused, but kept for potential future use).
        max_gpus (int): Maximum number of GPUs to select.
        min_vram (float): Minimum required free VRAM in gigabytes.

    Returns:
        List[int]: A list of GPU device IDs that meet the criteria.

    Raises:
        ValueError: If no GPUs are available or none have at least `min_vram` GB free.
    """
    if not torch.cuda.is_available():
        raise ValueError("No GPUs available.")
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024 ** 3)  # Convert to GB
        if free_gb >= min_vram:
            available_gpus.append((i, free_gb))
    
    if not available_gpus:
        raise ValueError(f"No GPUs with at least {min_vram} GB of free VRAM available.")
    
    available_gpus.sort(key=lambda x: x[1], reverse=True)
    selected_gpus = [gpu_id for gpu_id, mem in available_gpus[:max_gpus]]
    
    return selected_gpus
