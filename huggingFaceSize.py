from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()

for repo in cache_info.repos:
    print(
        f"Model: {repo.repo_id}",
        f"Size: {repo.size_on_disk/(1024**3):.2f} GB"
    )