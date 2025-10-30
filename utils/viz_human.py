
# python visulaize/smpl_run.py  --path /home/jinzhu/Downloads/amass_x_G1_frame/train/CMU/06/06_05_stageii.pkl --model_dir /home/jinzhu/Downloads/models/smplx
from isaacgym import gymapi, gymutil
import os
import argparse
import numpy as np
import time
import pickle

from smpl_sim.smpllib.smpl_parser import SMPLX_Parser
BACKEND = "smpl_sim"

# -------------------- 小工具 --------------------
def rot_matrix(axis, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == 'x': return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
    if axis == 'y': return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
    if axis == 'z': return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    raise ValueError("axis must be x/y/z")

def apply_rotate(X, rotate):
    if rotate == 'none': return X
    ax = rotate[0]
    sign = -1 if '-' in rotate else 1
    Rm = rot_matrix(ax, 90 * sign)
    return (X @ Rm.T)

# 常用的 24 体干骨架连线（只画身体，不画手脸）
SMPL_BODY24_PAIRS = [
    (0,3),(3,6),(6,9),(9,12),(12,15),   # pelvis->spine1->spine2->spine3->head
    (0,1),(1,4),(4,7),                  # left leg
    (0,2),(2,5),(5,8),                  # right leg
    (12,16),(16,18),                    # left shoulder->elbow
    (12,17),(17,19),                    # right shoulder->elbow
]

def _to_numpy(x):
    import torch
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if 'torch' in str(type(x)):
        # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

# -------------------- smpl_sim 前向封装（SMPL-X） --------------------
def smplx_forward(parser, pose_aa_np, betas_np, trans_np):
    """
    parser: SMPLX_Parser 实例
    输入 numpy：pose_aa (D,), betas (B,) or None, trans (3,)
    输出 numpy：joints (J,3), verts (V,3), faces (F,3)
    """
    import torch  # 在 isaacgym 之后导入，避免冲突
    device = getattr(parser, "device", "cpu")
    dtype = torch.float32
    pose_t = torch.as_tensor(pose_aa_np, dtype=dtype, device=device).reshape(1, -1)
    if betas_np is None:
        bet_t = torch.zeros((1, 10), dtype=dtype, device=device)
    else:
        bet_t  = torch.as_tensor(betas_np, dtype=dtype, device=device).reshape(1, -1)
    trn_t  = torch.as_tensor(trans_np, dtype=dtype, device=device).reshape(1, 3)

    ret = parser.get_joints_verts(pose_t, bet_t, trn_t)
    if isinstance(ret, (list, tuple)):
        if len(ret) == 3:
            verts_t, joints_t, _ = ret
        else:
            verts_t, joints_t = ret
    else:
        raise RuntimeError("Unexpected return of get_joints_verts")

    joints = joints_t[0].detach().cpu().numpy()
    verts  = verts_t[0].detach().cpu().numpy()
    faces  = getattr(parser, "faces", None)
    return joints, verts, faces

# -------------------- 数据加载：支持 .pkl / .npz --------------------
def load_sequence(path):
    """
    返回统一字典：
      poses: (T, D)  axis-angle concat
      trans: (T, 3)  或 None（若不存在则给 0）
      betas: (B,)    或 None
      fps:   float
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        data = _to_numpy(data)  # 把可能的 torch 张量转成 numpy

        # 兼容字段名
        poses = data.get("poses", None)
        if poses is None:
            # 如果没有 poses，也可由 root_orient + pose_body 拼
            root_orient = data.get("root_orient", None)   # (T,3)
            pose_body   = data.get("pose_body", None)     # (T,(J-1)*3)
            parts = []
            if root_orient is not None: parts.append(root_orient)
            if pose_body   is not None: parts.append(pose_body)
            if parts:
                poses = np.concatenate(parts, axis=1)
            else:
                raise ValueError("PKL missing 'poses' (or 'root_orient'+'pose_body').")

        trans = data.get("trans", None)
        betas = data.get("betas", None)

        # fps 兼容两种拼写
        fps = data.get("mocap_frame_rate", data.get("mocap_framerate", 60.0))
        fps = float(np.array(fps).reshape(-1)[0])

        return {
            "poses": poses.astype(np.float32),
            "trans": (np.zeros((poses.shape[0], 3), dtype=np.float32) if trans is None else trans.astype(np.float32)),
            "betas": (None if betas is None else betas.astype(np.float32).reshape(-1)),
            "fps": fps
        }

    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        poses = data["poses"]            # (T, D)
        trans = data.get("trans", None)  # (T, 3) or None
        betas = data.get("betas", None)
        fps = data.get("mocap_framerate", 60.0)
        try:
            fps = float(np.array(fps).reshape(-1)[0])
        except Exception:
            fps = 60.0
        return {
            "poses": poses.astype(np.float32),
            "trans": (np.zeros((poses.shape[0], 3), dtype=np.float32) if trans is None else trans.astype(np.float32)),
            "betas": (None if betas is None else betas.astype(np.float32).reshape(-1)),
            "fps": fps
        }
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def main():
    """
    可视化 .pkl（你导出的 AMASS/SMPL-X 同构字段）或 .npz（原 AMASS）
    """
    ap = argparse.ArgumentParser(description="IsaacGym: visualize a SMPL-X sequence (.pkl or .npz)")
    ap.add_argument("--path", type=str, required=True, help="路径到 .pkl 或 .npz")
    ap.add_argument("--model_dir", type=str, default=os.environ.get("SMPL_MODELS_DIR", None),
                    help="SMPL 模型目录（包含 SMPLX_*.pkl/npz）")
    ap.add_argument("--model_type", type=str, default="smplx", choices=["smplx"],
                    help="使用的模型类型（此版本固定 smplx）")
    ap.add_argument("--smpl_backend", type=str, default="smplx",
                    choices=["smplx", "custom"],
                    help="SMPL 后端解析方式")
    ap.add_argument("--betas_global", type=str, default=None,
                    help="可选的 shape 参数文件路径 (.npy / .npz)，否则使用文件自带的 betas")
    ap.add_argument("--rotate", type=str, default="none", choices=["none", "x90", "y90", "z90"],
                    help="是否对序列进行全局旋转")
    ap.add_argument("--skip", type=int, default=1, help="播放跳帧 (1=全帧)")
    ap.add_argument("--mode", type=str, default="skeleton", choices=["skeleton", "mesh"],
                    help="可视化模式 (骨架 skeleton / 网格 mesh)")
    ap.add_argument("--point_radius", type=float, default=0.02, help="骨架点半径")
    args = ap.parse_args()

    # 解析器：SMPL-X
    parser = SMPLX_Parser(
        model_path=args.model_dir,
        gender="NEUTRAL",
        ext="pkl",
        use_pca=False,
        num_pca_comps=0,
        num_betas=20
    )

    # 载入序列（pkl/npz 通吃）
    seq = load_sequence(args.path)
    poses = seq["poses"]     # (T, D)
    trans = seq["trans"]     # (T, 3)
    betas = seq["betas"]     # (B,) or None
    fps   = seq["fps"]       # float

    # 若用户指定全局 betas，则覆盖
    if args.betas_global is not None:
        if args.betas_global.endswith(".npy"):
            betas = np.load(args.betas_global).astype(np.float32).reshape(-1)
        else:
            _b = np.load(args.betas_global, allow_pickle=True)
            betas = (_b["betas"] if "betas" in _b else np.array(_b)).astype(np.float32).reshape(-1)

    T, D = poses.shape
    print(f"[SMPL-X] loaded {args.path} T={T}, D={D}, fps={fps}, has_trans={trans is not None}, betas_dim={None if betas is None else betas.shape[0]}")

    # ---------- Isaac Gym 初始化 ----------
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1.0/60.0
    sim_params.substeps = 1
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("create_sim 失败")

    # 地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # viewer 和 env
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("create_viewer 失败")
    env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)

    # 摄像机看向
    gym.viewer_camera_look_at(
        viewer, env,
        gymapi.Vec3(0.8, -2.2, 1.3),
        gymapi.Vec3(0.0,  0.0, 1.0)
    )

    # 颜色与几何
    COLOR_JOINT = gymapi.Vec3(0.1, 0.8, 0.2)
    COLOR_BONE  = (0.2, 0.6, 1.0)
    COLOR_MESH  = gymapi.Vec3(1.0, 0.5, 0.2)
    sphere_geom = gymutil.WireframeSphereGeometry(
        args.point_radius, 8, 8, color=(COLOR_JOINT.x, COLOR_JOINT.y, COLOR_JOINT.z)
    )

    # 播放控制
    t_index = 0
    real_dt = 1.0 / max(1.0, fps) * max(1, args.skip)
    last_time = time.time()
    paused = False

    print("[INFO] 播放：空格暂停/继续，左/右箭头快退/快进，Esc 关闭")

    while not gym.query_viewer_has_closed(viewer):
        gym.step_graphics(sim)
        now = time.time()
        if not paused and (now - last_time) >= real_dt:
            last_time = now

            pose_t = poses[t_index].astype(np.float32)
            trans_t = trans[t_index].astype(np.float32) if trans is not None else np.zeros(3, dtype=np.float32)
            bet_t = betas.astype(np.float32) if (betas is not None) else None

            joints, verts, faces = smplx_forward(parser, pose_t, bet_t, trans_t)

            if args.rotate != "none":
                joints = apply_rotate(joints, args.rotate)
                if verts is not None:
                    verts = apply_rotate(verts, args.rotate)

            gym.clear_lines(viewer)
            # 绘制 joints
            for j in range(joints.shape[0]):
                p = gymapi.Vec3(float(joints[j,0]), float(joints[j,1]), float(joints[j,2]))
                tf = gymapi.Transform(p=p, r=gymapi.Quat(0,0,0,1))
                gymutil.draw_lines(sphere_geom, gym, viewer, env, tf)

            # 画骨架线
            color = gymapi.Vec3(*COLOR_BONE)
            for (ia, ib) in SMPL_BODY24_PAIRS:
                if ia < joints.shape[0] and ib < joints.shape[0]:
                    a = joints[ia]; b = joints[ib]
                    start = gymapi.Vec3(float(a[0]), float(a[1]), float(a[2]))
                    end   = gymapi.Vec3(float(b[0]), float(b[1]), float(b[2]))
                    gymutil.draw_line(start, end, color, gym, viewer, env)

            t_index += args.skip
            if t_index >= T:
                t_index = 0   # 循环播放

        else:
            gym.draw_viewer(viewer, sim, True)
            time.sleep(0.001)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
