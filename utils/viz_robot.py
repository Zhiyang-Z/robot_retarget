from isaacgym import gymapi, gymutil, gymtorch
import os, pickle, numpy as np, torch

def viz_robot(urdf, root_pos, root_rot, dof_pos, fps=30.0, render=False):
    use_root = True # not bool(getattr(args, "no_root", False))
    T, D_in = dof_pos.shape

    # --- sim setup (no gravity for stability) ---
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0/60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.002
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.default_buffer_size_multiplier = 5.0
    sim_params.gravity = gymapi.Vec3(0,0,0)  # 关闭重力，纯可视化更稳

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None: raise RuntimeError("create_sim failed")
    # viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # if viewer is None: raise RuntimeError("create_viewer failed")

    # ground
    plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0,0,1)
    plane.static_friction = plane.dynamic_friction = 1.0
    gym.add_ground(sim, plane)

    # asset
    opts = gymapi.AssetOptions()
    opts.fix_base_link = not use_root       # 不用 root 时固定底座
    opts.disable_gravity = True             # 与 sim_params.gravity 对齐
    opts.armature = 0.01
    asset_root, asset_file = os.path.split(urdf)
    asset = gym.load_asset(sim, asset_root if asset_root else ".", asset_file, opts)
    if asset is None: raise RuntimeError(f"load_asset failed: {urdf}")

    D_model = gym.get_asset_dof_count(asset)
    T = dof_pos.shape[0]

    # env + actor（单环境）
    env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)
    init_pose = gymapi.Transform(); init_pose.p = gymapi.Vec3(0,0,0)  
    actor = gym.create_actor(env, asset, init_pose, "robot", 0, 1)

    # DOF limits（用来 clip）
    dof_props = gym.get_asset_dof_properties(asset)
    has_lim = ("lower" in dof_props.dtype.names) and ("upper" in dof_props.dtype.names)
    if has_lim:
        lo = dof_props["lower"].astype(np.float32)
        hi = dof_props["upper"].astype(np.float32)

    # buffers
    dof_states = np.zeros(D_model, dtype=gymapi.DofState.dtype)
    if use_root and not opts.fix_base_link:
        root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1,13)
        actor_idx = torch.tensor([gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)],
                                 dtype=torch.int32, device=root_states.device)

    # camera
    # gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.8,1.8,1.2), gymapi.Vec3(0,0,0.7))
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    cam_props.enable_tensors = False
    camera_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(1.8,1.8,1.2), gymapi.Vec3(0,0,0.7))

    # --- init to frame 0 ---
    q0 = dof_pos[0];  q0 = np.clip(q0, lo, hi) if has_lim else q0
    dof_states["pos"] = q0; dof_states["vel"] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

    if use_root and not opts.fix_base_link:
        px, py, pz = root_pos[0]
        q_xyzw = root_rot[0]
        qn = np.linalg.norm(q_xyzw);  q_xyzw = q_xyzw / max(1e-12, qn)
        root_states[actor_idx, 0:3]  = torch.tensor([px,py,pz], dtype=torch.float32, device=root_states.device)
        root_states[actor_idx, 3:7]  = torch.tensor(q_xyzw,   dtype=torch.float32, device=root_states.device)
        root_states[actor_idx, 7:13] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))

    print(f"URDF: {urdf} | dofs={D_model} | frames={T} | fps={fps:.2f} | use_root={use_root}")

    # --- playback ---
    dt = sim_params.dt
    frame_dt = 1.0/max(1.0, fps)
    accum = 0.0
    frame = 0
    import imageio.v2 as imageio
    video = np.zeros((T-1, cam_props.height, cam_props.width, 3), dtype=np.uint8)

    while frame < T-1:
        # 累积时间，按 fps 推进索引
        accum += dt
        advanced = False
        while accum >= frame_dt:
            frame = (frame + 1) # % T
            accum -= frame_dt
            advanced = True

        if advanced:
            q = dof_pos[frame]; q = np.clip(q, lo, hi) if has_lim else q
            dof_states["pos"] = q; dof_states["vel"] = 0.0
            gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

            if use_root and not opts.fix_base_link:
                px, py, pz = root_pos[frame]
                q_xyzw = root_rot[frame]
                root_states[actor_idx, 0:3]  = torch.tensor([px,py,pz], dtype=torch.float32, device=root_states.device)
                root_states[actor_idx, 3:7]  = torch.tensor(q_xyzw,   dtype=torch.float32, device=root_states.device)
                root_states[actor_idx, 7:13] = 0.0
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        # gym.draw_viewer(viewer, sim)
        gym.sync_frame_time(sim)
        #####
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        image_tensor = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
        image_tensor = image_tensor.reshape(cam_props.height, cam_props.width, 4)[..., :3]
        video[frame - 1] = image_tensor

    # save video
    # imageio.mimsave('output.mp4', video, fps=fps)
    return video


    # gym.destroy_viewer(viewer); gym.destroy_sim(sim)

if __name__ == "__main__":
    # # --- load data ---
    with open("/home/zhiyang/projects/robot/clip/01_01_stageii_robot.pkl", "rb") as f:
        data = pickle.load(f)
    print(data.keys())
    dof_pos  = np.asarray(data["dof_pos"],  np.float32)      # [T, D]
    print(dof_pos.shape)
    root_pos = np.asarray(data["root_pos"], np.float32)      # [T, 3]
    root_rot = np.asarray(data["root_rot"], np.float32)      # [T, 4]  <-- 必须是 xyzw！
    # fps      = float(data.get("fps", 30.0))

    viz_robot("/home/zhiyang/projects/robot/G1/g1_29dof.urdf", root_pos, root_rot, dof_pos)