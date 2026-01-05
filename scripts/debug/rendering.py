from remo_splat.configs import ExampleBookshelf
from remo_splat.utils import Render

config = ExampleBookshelf(hist = False)
config_2d = ExampleBookshelf(is_3D=False, hist = False)

render_3D = Render(config)

render_2D = Render(config_2d)

render_3D.sample_render([1,2])
render_2D.sample_render([1,2])

__import__('pdb').set_trace()
