
from roboflow import Roboflow
rf = Roboflow(api_key="29cYOOTGlTb1nYSVgUVq")
project = rf.workspace("gn-nhn-yc6af").project("new-b6snk")
version = project.version(1)
dataset = version.download("coco")
                