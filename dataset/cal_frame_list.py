from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

lowest_frame = 64

top_path = Path(f"/media/zhiyang/zhiyang_HDD/robot/")
human_path, robot_path = top_path / "amass_x_G1_human/train/", top_path / "amass_new_robot/"
human_trajs = sorted(list(human_path.glob("**/*_target14_rotvec.pkl")), key=lambda x: x.name)
robot_trajs = sorted(list(robot_path.glob("**/*.pkl")), key=lambda x: x.name)

frames = []
minimal_frame = 10000
delete_file = []
for h_traj in tqdm(human_trajs):
    with open(h_traj, "rb") as f:
        h_data = pickle.load(f)
    h_pose = h_data['poses_body_sel']

    if h_pose.shape[0] < lowest_frame:
        delete_file.append(h_traj)
        continue

    with open(Path(str(h_traj).replace("amass_x_G1_human/train", "amass_new_robot").replace("_target14_rotvec", "")), "rb") as f:
        r_data = pickle.load(f)
    r_pose = r_data["dof_pos"]
    assert h_pose.shape[0] == r_pose.shape[0]
    frames.append(h_pose.shape[0] - lowest_frame + 1)
    if h_pose.shape[0] < minimal_frame: minimal_frame = h_pose.shape[0]

# print(frames)
accum_frames = np.cumsum(np.array(frames))
np.save("./frame_list", accum_frames)
print(minimal_frame)
with open("delete_files.pkl", "wb") as f:
    pickle.dump(delete_file, f)