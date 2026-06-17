from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="outputs/weights/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id="Khalil200383/UrbanSeg-AI",
    repo_type="model"
)



