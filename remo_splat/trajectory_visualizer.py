from typing import List, Optional, Tuple

import numpy as np
import spatialgeometry as sg
import spatialmath as sm
import swift
from mm_neo.ideal_sdf import load_from_json
from neural_robot.unity_frankie import NeuralFrankie

from remo_splat import logger


class TrajectoryReplayer:
    def __init__(self, exp_names: List[str], colors: List[np.ndarray]):
        """
        Visualizer to compare different experiments runs

        Args:
            exp_names: List[str] name of the folder to load under ./logs/experiments
        """
        self.env = swift.Swift()
        self.env.launch(True, browser="chromium", rate=1, headless=False)

        self.instances = [logger.load_folder(exp_name) for exp_name in exp_names]
        self.robots = [NeuralFrankie() for _ in self.instances]
        for ix, robot in enumerate(self.robots):
            robot.paint(colors[ix])
            self.env.add(robot)

        self.data = [logger.LoggerLoader(instance[0], "", "") for instance in self.instances]
        self.max_steps = 200
        self.n_instances = len(self.instances)

        self.step_slider = swift.Slider(lambda x: self.step(x), 0, 200, 1, value= 0, desc ="step") 
        self.env.add(self.step_slider)

        self.step_episode_slider = swift.Slider(lambda x:self.step_episode(x), 0, 200, 1, value= 0, desc ="step_episode") 
        self.env.add(self.step_episode_slider)

        _, self.swift_obstacles, _ = load_from_json(self.data[0].data_folder + "scene.json")
        for obs in self.swift_obstacles:
            obs.color = np.random.randint(0, 255, (3))
            self.env.add(obs)


    def step_robot(self, step_id, robot_id):
        q = self.data[robot_id].get_data("q")
        T_WB = self.data[robot_id].get_data("T_WB")
        step = min(step_id, q.shape[0] -1)
        self.robots[robot_id].q = q[step]
        self.robots[robot_id].base = sm.SE3(T_WB[step])

    def step(self, id):
        for i in range(self.n_instances):
            self.step_robot(id, i)
        self.env.step(0.01)
        

    def step_episode(self, id):

        self.data = [logger.LoggerLoader(instance[id], "", "") for instance in self.instances]
        _, obstacles, _ = load_from_json(self.data[0].data_folder + "scene.json")
        for ix, ob in enumerate(obstacles):
            self.swift_obstacles[ix].T = ob.T
            # TODO: Update the sizes also
            # if (isinstance(ob, sg.Cuboid)):
                # self.swift_obstacles[ix].scale = ob.scale
            # elif (isinstance(ob, sg.Cylinder)):
                # self.swift_obstacles[ix].radius = ob.radius
                # self.swift_obstacles[ix].length = ob.length


        self.step(0)

        

if __name__ == "__main__":
    from dataclasses import dataclass, field

    import tyro

    @dataclass
    class Config:
        exp_names: List[str] = field(default_factory=lambda: ["final_rev2/bookshelf/2D/depth", "final_rev2/bookshelf/2D/depthactive"])
        "List of names of folder under .logs/experiments"
        colors: Optional[List[List]] = field(default_factory= lambda: [[1,1,0],[0,0,1]] )
        "List of r,g,b colors for differentiate the robots"
    
    args = tyro.cli(Config)
    if not args.colors:
        args.colors = [ np.random.randn(3) for _ in range(len(args.exp_names))]
    replayer =TrajectoryReplayer(args.exp_names, args.colors)
    while True:
        replayer.env.step()


