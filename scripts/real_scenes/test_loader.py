import swift
import tyro

from remo_splat.configs.experiments import ReachingRealWorldConfig
from remo_splat.swift_utils import draw_cameras,draw_target_poses

if __name__ =="__main__":
    args = tyro.cli(ReachingRealWorldConfig)
    env = swift.Swift()
    env.launch(browser = "chromium")
    print(args)
    draw_target_poses(env, args.T_WEp)
    draw_cameras(env, args.T_WC)
    env.hold()
