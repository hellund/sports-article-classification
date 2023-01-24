from pathlib import Path


def get_project_root() -> str:
    """Get the root of the project folder

    Returns:
        str: str containing the root of the project folder
    """
    root = Path(__file__).parent.parent
    root = str(root).replace('\\', '/')
    return root
