"""Static collision geometry used only by the sensor simulator and checks."""

import math
import xml.etree.ElementTree as ET


def compose_pose(parent, text):
    local = list(map(float, text.split()))
    x, y, z, yaw = parent
    if abs(local[3]) > 1e-8 or abs(local[4]) > 1e-8:
        raise ValueError("Static visibility checks require upright collision shapes")
    return (x + math.cos(yaw)*local[0] - math.sin(yaw)*local[1],
            y + math.sin(yaw)*local[0] + math.cos(yaw)*local[1],
            z + local[2], yaw + local[5])


def load_obstacles(path, min_top_m=0.10):
    shapes = []
    for model in ET.parse(path).getroot().find("world").findall("model"):
        if model.findtext("static") != "true":
            continue
        model_pose = compose_pose((0, 0, 0, 0), model.findtext("pose", "0 0 0 0 0 0"))
        for link in model.findall("link"):
            link_pose = compose_pose(model_pose, link.findtext("pose", "0 0 0 0 0 0"))
            for collision in link.findall("collision"):
                x, y, z, yaw = compose_pose(link_pose, collision.findtext("pose", "0 0 0 0 0 0"))
                geometry = collision.find("geometry")
                box = geometry.findtext("box/size")
                cylinder = geometry.find("cylinder")
                if box is not None:
                    sx, sy, sz = map(float, box.split())
                    if z + sz/2 > min_top_m:
                        shapes.append(("box", x, y, yaw, sx/2, sy/2))
                elif cylinder is not None:
                    if z + float(cylinder.findtext("length"))/2 > min_top_m:
                        shapes.append(("cylinder", x, y, 0, float(cylinder.findtext("radius")), 0))
                else:
                    raise ValueError("Unsupported static collision shape in visibility map")
    return shapes


def line_of_sight(start, end, shapes):
    """Return False if the segment intersects a static box or cylinder."""
    for kind, x, y, yaw, a, b in shapes:
        c, s = math.cos(yaw), math.sin(yaw)
        local = [(c*(p[0]-x)+s*(p[1]-y), -s*(p[0]-x)+c*(p[1]-y))
                 for p in (start, end)]
        p, q = local
        if kind == "cylinder":
            dx, dy = q[0]-p[0], q[1]-p[1]
            length2 = dx*dx + dy*dy
            t = 0.0 if length2 == 0 else max(0, min(1, -(p[0]*dx+p[1]*dy)/length2))
            if math.hypot(p[0]+t*dx, p[1]+t*dy) <= a:
                return False
        else:
            near, far = 0.0, 1.0
            for axis, half in enumerate((a, b)):
                delta = q[axis]-p[axis]
                if abs(delta) < 1e-12:
                    if abs(p[axis]) > half:
                        near, far = 1.0, 0.0
                        break
                else:
                    lo, hi = sorted(((-half-p[axis])/delta, (half-p[axis])/delta))
                    near, far = max(near, lo), min(far, hi)
            if near <= far:
                return False
    return True
