import numpy as np
import pandas as pd

from waste_classifier.near_duplicates import flag_near_duplicates, nearest_rmse


def test_nearest_rmse_finds_the_closest_image():
    pool = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    queries = np.array([[0.9, 0.9]], dtype=np.float32)
    assert abs(nearest_rmse(queries, pool)[0] - 0.1) < 1e-3


def test_flags_depend_on_which_images_a_model_could_have_trained_on():
    df = pd.DataFrame(
        {
            "domain": ["default", "real_world", "default", "real_world", "real_world"],
            "split": ["train", "train", "test", "test", "val"],
        }
    )
    thumbnails = np.array(
        [
            [0.0, 0.0],  # studio training image
            [1.0, 1.0],  # real_world training image
            [0.0, 0.0],  # copy of the studio image
            [1.0, 1.0],  # copy of the real_world image only
            [0.5, 0.0],  # unique
        ],
        dtype=np.float32,
    )

    flagged = flag_near_duplicates(df, thumbnails, threshold=0.03)

    assert flagged["near_studio"].tolist() == [False, False, True, False, False]
    assert flagged["near_any"].tolist() == [False, False, True, True, False]
