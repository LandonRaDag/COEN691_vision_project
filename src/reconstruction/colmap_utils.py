"""
COLMAP reconstruction utilities.

Handles Structure-from-Motion and Multi-View Stereo reconstruction.
"""

def run_sfm(image_dir: str, output_dir: str):
    """
    Run COLMAP Structure-from-Motion to estimate camera poses.

    Parameters
    ----------
    image_dir : str
        Directory of input images.
    output_dir : str
        Directory where COLMAP sparse reconstruction will be stored.
    """

    raise NotImplementedError("COLMAP SfM pipeline not implemented yet.")


def run_dense_reconstruction(workspace_dir: str):
    """
    Run COLMAP dense reconstruction to generate a point cloud.

    Parameters
    ----------
    workspace_dir : str
        COLMAP workspace directory containing sparse model.
    """

    raise NotImplementedError("COLMAP dense reconstruction not implemented.")