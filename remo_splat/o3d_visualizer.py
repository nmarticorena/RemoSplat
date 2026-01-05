import threading
import time
from collections import defaultdict

import numpy as np
import open3d as o3d
from open3d.t.geometry import PointCloud as TensorPointCloud
from open3d.visualization import gui, rendering

"""TODO list
[x] Have lock to avoid datarace
[ ] Improve init_render to be based on the current scene
[ ] Add set camera method
[x] Improve point size information
"""

# States
UNCHANGED = 0
NEW_GEOMETRY = 1
UPDATE_GEOMETRY = 2


def to_tensor_point_cloud(geometry):
    """Convert an `open3d.geometry.PointCloud` to `open3d.t.geometry.PointCloud`."""
    tensor_pcd = TensorPointCloud()
    tensor_pcd.point.positions = o3d.core.Tensor(
        np.asarray(geometry.points), dtype=o3d.core.Dtype.Float32
    )
    if geometry.colors:
        tensor_pcd.point.colors = o3d.core.Tensor(
            np.asarray(geometry.colors), dtype=o3d.core.Dtype.Float32
        )
    if geometry.normals:
        tensor_pcd.point.normals = o3d.core.Tensor(
            np.asarray(geometry.normals), dtype=o3d.core.Dtype.Float32
        )
    return tensor_pcd


class Visualizer:
    def __init__(self, web=False):
        if web:
            o3d.visualization.webrtc_server.enable_webrtc()
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = gui.Application.instance.create_window("Visualizer", 1920, 1080)

        self.window.set_on_layout(self._on_layout)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.window.add_child(self.scene)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.sRGB_color = True

        self.unlit_material = rendering.MaterialRecord()
        self.unlit_material.shader = "unlitLine"
        self._changed = False
        self.is_done = False
        self.lock = threading.Lock()
        self.geometries = defaultdict()
        self.geometries_changed: dict[str, int] = defaultdict(int)  # 1 new, 2 update
        self.geometries_material: dict[str, rendering.MaterialRecord] = defaultdict(
            rendering.MaterialRecord
        )

        w = self.window
        em = w.theme.font_size

        # Add panels to the window
        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(spacing, spacing, spacing, spacing)
        self.spacing = spacing
        self.panel = gui.Vert(0, margins)
        w.add_child(self.panel)

        # Panel with the controls
        gui.Application.instance.post_to_main_thread(self.window, self._on_start)
        threading.Thread(name="UpdateMain", target=self.update_main).start()

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.scene.frame = gui.Rect(x, rect.y, rect.get_right() - x, rect.height)

    def get_mat(
        self, point_size: int = 1, color=[1.0, 0, 0, 1], line=False
    ) -> rendering.MaterialRecord:
        """Helper to do not require to import open3d"""
        mat = rendering.MaterialRecord()
        if line:
            mat.shader = "unlitLine"
            mat.line_width = point_size
        else:
            mat.shader = "defaultUnlit"
        mat.sRGB_color = True
        mat.point_size = point_size
        mat.base_color = np.array(color)
        return mat

    def init_render(self):
        self.window.set_needs_layout()
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.scene.setup_camera(60, bbox, [0, 0, 0])
        self.scene.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def _on_start(self):
        self.init_render()

    # Major loop
    def update_main(self):
        while not self.is_done:
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render()
            )
            time.sleep(0.1)
            pass

    def update_render(self):
        if self._changed:
            with self.lock:
                for k, v in self.geometries_changed.items():
                    mat = self.geometries_material[k]
                    if v == 1:
                        self.scene.scene.scene.add_geometry(k, self.geometries[k], mat)
                        self.geometries_changed[k] = UNCHANGED
                    elif v == 2:
                        self.scene.scene.scene.remove_geometry(k)
                        self.scene.scene.scene.add_geometry(k, self.geometries[k], mat)
                        self.geometries_changed[k] = UNCHANGED
                self._changed = False

    def add_geometry(self, name, geometry, material=None):
        with self.lock:
            # if geometry is o3d.geometry.PointCloud:
            #     print("TensorPointCloud")
            #     self.geometries[name] = to_tensor_point_cloud(geometry)
            # else:
            self.geometries[name] = geometry
            if material is None:
                if isinstance(geometry, o3d.geometry.LineSet):
                    material = self.unlit_material
                else:
                    material = self.mat
            self.geometries_material[name] = material
            self._changed = True
            self.geometries_changed[name] = 1

    def update_material(self, name, material):
        with self.lock:
            if name in self.geometries_material:
                self.geometries_material[name] = material
                self._changed = True
            else:
                raise ValueError(f"Geometry {name} not found in the visualizer.")

    def update_geometry(self, name, geometry):
        with self.lock:
            if geometry is o3d.geometry.PointCloud:
                self.geometries[name] = to_tensor_point_cloud(geometry)
            else:
                self.geometries[name] = geometry
            self._changed = True
            self.geometries_changed[name] = 2

    def set_camera(self, camera_info: str):
        """Set camera info from json file"""
        original = self.ctr.convert_to_pinhole_camera_parameters()
        self.extrinsic = o3d.io.read_pinhole_camera_parameters(camera_info).extrinsic

    def add_slider(self, name, min_value, max_value, on_value_changed, is_float: bool = False) -> gui.Slider:
        label = gui.Label(name)
        if is_float:
            slider = gui.Slider(gui.Slider.DOUBLE)
        else:
            slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(min_value, max_value)
        slider.set_on_value_changed(on_value_changed)
        self.panel.add_child(label)
        self.panel.add_child(slider)
        return slider

    def add_checkbox(self, name, on_value_changed, default: bool = False):
        checkbox = gui.Checkbox(name)
        checkbox.checked = default
        checkbox.set_on_checked(on_value_changed)
        self.panel.add_child(checkbox)
        return checkbox

    def add_button(self, name, set_on_clicked):
        button = gui.Button(name)
        button.set_on_clicked(set_on_clicked)
        self.panel.add_child(button)
        return button

    def add_radio_button(self, items, on_checked_changed, default_index: int = 0):
        radio = gui.RadioButton(gui.RadioButton.Type.HORIZ)  # Horizontal
        radio.set_items(items)
        radio.set_on_selection_changed(on_checked_changed)
        self.panel.add_child(radio)
        radio.selected_index = default_index
        # on_checked_changed(default_index)
        return radio


if __name__ == "__main__":
    from remo_splat.configs import ExampleReplica as config

    dataset = config().dataset_helper

    o3d.visualization.webrtc_server.enable_webrtc()
    vis = Visualizer()

    first_frame = dataset.sample_pcd(0)
    vis.add_geometry("pcd", first_frame)
    j = 0
    while True:
        vis.update_geometry("pcd", dataset.sample_pcd(j))
        j += 1
        j = j % dataset.n_frames
        for _ in range(100):
            app.run_one_tick()
