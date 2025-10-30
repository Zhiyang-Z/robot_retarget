from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import pickle

class RetargetDataset(Dataset):
    def __init__(self, path, delete_file_path, frame_list_path, n_frame=64, frame_skip=1):
        self.n_frame, self.frame_skip = n_frame, frame_skip
        top_path = Path(f"{path}")
        human_path, robot_path = top_path / "amass_x_G1_human/train/", top_path / "amass_new_robot/"
        self.human_trajs = sorted(list(human_path.glob("**/*_target14_rotvec.pkl")), key=lambda x: x.name)
        self.robot_trajs = sorted(list(robot_path.glob("**/*.pkl")), key=lambda x: x.name)
        print(f"Before drop out: {len(self.human_trajs)}")
        # drop out too short human_traj
        with open(delete_file_path, "rb") as f:
            delete_list = pickle.load(f)
        self.human_trajs = [x for x in self.human_trajs if x not in delete_list]
        self.human_trajs = sorted(self.human_trajs, key=lambda x: x.name)
        print(f"After drop out: {len(self.human_trajs)}")

        self.accum_frames_list = np.load(frame_list_path)
        assert self.accum_frames_list.shape[0] == len(self.human_trajs), f"{self.accum_frames_list.shape[0]} is not equal to {len(self.human_trajs)}"

    def __len__(self):
        return self.accum_frames_list[-1]

    def __getitem__(self, idx):
        file_idx = np.argmax(self.accum_frames_list > idx)
        clip_idx = idx - np.pad(self.accum_frames_list, (1, 0))[file_idx]
        cur_path = self.human_trajs[file_idx]
        with open(cur_path, "rb") as f:
            h_data = pickle.load(f)
        h_pose = h_data['poses_body_sel']
        with open(Path(str(self.human_trajs[file_idx]).replace("amass_x_G1_human/train", "amass_new_robot").replace("_target14_rotvec", "")), "rb") as f:
            r_data = pickle.load(f)
        r_pose = r_data["dof_pos"]
        assert h_pose.shape[0] == r_pose.shape[0] and h_pose.shape[1] == 42 and r_pose.shape[1] == 29
        root_pos = r_data["root_pos"]
        root_rot = r_data["root_rot"]

        return h_pose[clip_idx: (clip_idx+self.n_frame), :].astype(np.float32), r_pose[clip_idx: (clip_idx+self.n_frame), :].astype(np.float32), \
            root_pos[clip_idx: (clip_idx+self.n_frame), :].astype(np.float32), root_rot[clip_idx: (clip_idx+self.n_frame), :].astype(np.float32)
    
if __name__ == "__main__":
    from tqdm import tqdm
    dataset = RetargetDataset(f"/media/zhiyang/zhiyang_HDD/robot/", f"/home/zhiyang/projects/robot/robot_retarget/dataset/frame_list.npy")
    for data in tqdm(dataset):
        pass