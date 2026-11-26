"""Build a 20-second clarinet assembly animation in Blender; see README for usage."""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_walk_in_field import (MeshBuilder, catmull_rom, finger, profile_at,  # noqa: E402
                                sgnpow, smoothstep, stations)

FPS = 30
FRAMES = 20 * FPS

# The character faces -Y. The upper body leans about CHEST so the hands can reach the case.
CHEST = Vector((0, 0.012, 0.78))
HEAD_PIVOT = Vector((0, -0.020, 1.215))
UPPER_ARM, FOREARM = 0.285, 0.250
REACH = 0.97 * (UPPER_ARM + FOREARM)
GRIP = 0.027                     # palm centre to the axis of the gripped tube
TORSO_SHAPE = 2.3
TORSO = catmull_rom([            # z, half-depth, half-width, y-centre
    (0.660, 0.072, 0.128, 0.020),
    (0.720, 0.082, 0.138, 0.014),
    (0.800, 0.084, 0.126, 0.008),
    (0.860, 0.087, 0.129, 0.005),
    (0.920, 0.091, 0.140, 0.004),
    (0.970, 0.092, 0.150, 0.006),
    (1.010, 0.087, 0.158, 0.010),
    (1.035, 0.078, 0.160, 0.014),
    (1.060, 0.064, 0.140, 0.016),
    (1.085, 0.052, 0.094, 0.017),
    (1.105, 0.046, 0.058, 0.018),
    (1.140, 0.043, 0.047, 0.019),
    (1.190, 0.041, 0.044, 0.022),
], 10)
# Distance from the shoulder joint, radius: puffed sleeve, cuff, upper arm, forearm, wrist.
ARM_KEYS = [(0.000, 0.050), (0.025, 0.058), (0.070, 0.063), (0.115, 0.060), (0.150, 0.054),
            (0.156, 0.050), (0.160, 0.051), (0.166, 0.049), (0.170, 0.040), (0.200, 0.039),
            (0.250, 0.036), (0.285, 0.034), (0.315, 0.036), (0.360, 0.034), (0.430, 0.028),
            (0.500, 0.022), (0.535, 0.021)]


def material(name, color, roughness=0.55, metallic=0.0, bump=0.0, sss=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    shader = nt.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if sss:
        shader.inputs["Subsurface Weight"].default_value = sss
        shader.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        shader.inputs["Subsurface Scale"].default_value = 0.01
    if bump:
        coord = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 260
        noise.inputs["Detail"].default_value = 4
        relief = nt.nodes.new("ShaderNodeBump")
        relief.inputs["Strength"].default_value = bump
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], relief.inputs["Height"])
        nt.links.new(relief.outputs["Normal"], shader.inputs["Normal"])
    mat.diffuse_color = (*color, 1)
    return mat


def hair_material():
    mat = material("Dark chocolate hair", (0.058, 0.024, 0.015), 0.36)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    strands = nt.nodes.new("ShaderNodeTexWave")
    strands.bands_direction = "X"
    strands.inputs["Scale"].default_value = 90
    strands.inputs["Distortion"].default_value = 6
    strands.inputs["Detail"].default_value = 3
    relief = nt.nodes.new("ShaderNodeBump")
    relief.inputs["Strength"].default_value = 0.35
    nt.links.new(coord.outputs["Object"], strands.inputs["Vector"])
    nt.links.new(strands.outputs["Fac"], relief.inputs["Height"])
    nt.links.new(relief.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    return mat


def finish(name, mat, parent=None, smooth=False):
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.data.materials.append(mat)
    if smooth:
        for face in obj.data.polygons:
            face.use_smooth = True
    return obj


def pivot(name, location=(0, 0, 0), parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    return obj


def attach(obj, parent, origin):
    """Parent an object modelled in world space to a pivot that sits at `origin`."""
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Translation(origin).inverted()
    return obj


def box(name, location, size, mat, bevel=0.015, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = finish(name, mat, parent)
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    edge = obj.modifiers.new("Rounded edges", "BEVEL")
    edge.width = bevel
    edge.segments = 4
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def oval(name, location, radii, mat, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, location=location)
    obj = finish(name, mat, parent, smooth=True)
    obj.scale = radii
    return obj


def profile(name, rings, mat, parent=None):
    count = 40
    verts = [(r * math.cos(i * math.tau / count), r * math.sin(i * math.tau / count), z)
             for z, r in rings for i in range(count)]
    faces = [(j * count + i, j * count + (i + 1) % count,
              (j + 1) * count + (i + 1) % count, (j + 1) * count + i)
             for j in range(len(rings) - 1) for i in range(count)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    mesh.materials.append(mat)
    for face in mesh.polygons:
        face.use_smooth = True
    return obj


def line(name, points, radius, mat, parent=None, closed=False):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, point in zip(spline.bezier_points, points):
        bp.co = point
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    curve.materials.append(mat)
    return obj


# ---------------------------------------------------------------- character
def shoulder_rest(side):
    return Vector((side * 0.150, 0.012, 0.995))


def torso_dims(z):
    return profile_at(TORSO, z)


def torso_point(z, phi):
    a, b, cy = torso_dims(z)
    e = 2 / TORSO_SHAPE
    return Vector((b * sgnpow(math.sin(phi), e), cy - a * sgnpow(math.cos(phi), e), z))


def neckline(front):
    """Height of the scoop neckline; `front` is 1 at the centre front, 0 at the sides."""
    return 1.075 - 0.040 * front * front


def build_upper_body(chest, skin, shirt, trim):
    SKIN, SHIRT = range(2)
    zs = stations(TORSO[0][0], TORSO[-1][0], 0.012)
    dims = [torso_dims(z) for z in zs]
    mb = MeshBuilder()
    mb.loft([(0, cy, z) for z, (_, _, cy) in zip(zs, dims)], [(a, b) for a, b, _ in dims],
            ring=40, ref=(0, 1, 0), shape=TORSO_SHAPE, caps=(0.5, 0.4))
    body = mb.build("Torso and neck", [skin, shirt])
    for v in body.data.vertices:                      # soft bust shaped into the chest
        x, y, z = v.co
        a, _, cy = torso_dims(min(max(z, zs[0]), zs[-1]))
        front = max(0.0, (cy - y) / a)
        bump = sum(math.exp(-((x - s * 0.056) / 0.042) ** 2 - ((z - 0.942) / 0.040) ** 2)
                   for s in (-1, 1))
        v.co.y -= 0.019 * bump * front * front
    for poly in body.data.polygons:
        c = poly.center
        a, _, cy = torso_dims(min(max(c.z, zs[0]), zs[-1]))
        front = min(1.0, max(0.0, (cy - c.y) / a))
        poly.material_index = SKIN if c.z > neckline(front) else SHIRT
    attach(body, chest, CHEST)

    e = 2 / TORSO_SHAPE
    trim_points = []
    for k in range(36):
        phi = k * math.tau / 36
        p = torso_point(neckline(max(0.0, sgnpow(math.cos(phi), e))), phi)
        trim_points.append(p + (p - Vector((0, torso_dims(p.z)[2], p.z))).normalized() * 0.002)
    attach(line("Scoop neckline trim", trim_points, 0.0035, trim, closed=True), chest, CHEST)


def build_skirt(mat):
    # Cross-sections run from the waistband over the lap, then fall toward the ankles.
    rows = (
        ((0, 0.010, 0.775), (0.139, 0.092), 0.0),
        ((0, 0.012, 0.730), (0.150, 0.100), 0.0),
        ((0, 0.012, 0.660), (0.172, 0.118), 0.0),
        ((0, -0.035, 0.58), (0.193, 0.126), -0.40),
        ((0, -0.20, 0.555), (0.211, 0.103), -1.30),
        ((0, -0.392, 0.535), (0.222, 0.095), -1.40),
        ((0, -0.450, 0.425), (0.220, 0.100), -0.55),
        ((0, -0.463, 0.215), (0.222, 0.109), 0.0),
        ((0, -0.456, 0.123), (0.230, 0.114), 0.0),
    )
    count = 64
    verts = []
    for index, (center, (rx, ry), tilt) in enumerate(rows):
        for i in range(count):
            a = i * math.tau / count
            pleat = max(0.0, index - 2) * 0.0016 * math.cos(12 * a + index * 0.15)
            y = (ry + pleat) * math.sin(a)
            verts.append((center[0] + (rx + pleat) * math.cos(a),
                          center[1] + y * math.cos(tilt),
                          center[2] + y * math.sin(tilt)))
    faces = [(j * count + i, (j + 1) * count + i,
              (j + 1) * count + (i + 1) % count, j * count + (i + 1) % count)
             for j in range(len(rows) - 1) for i in range(count)]
    mesh = bpy.data.meshes.new("Pleated skirt")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("Ankle-length black skirt", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(mat)
    for face in mesh.polygons:
        face.use_smooth = True
    obj.modifiers.new("Drape", "SUBSURF")
    obj.modifiers.new("Cloth thickness", "SOLIDIFY").thickness = 0.003
    band = MeshBuilder()
    band.loft([(0, 0.010, 0.760), (0, 0.010, 0.788)], [(0.094, 0.141)] * 2, ring=48,
              ref=(0, 1, 0), caps=(0, 0))
    band.build("Skirt waistband", [mat])


def build_feet(skin, leather):
    for side in (-1, 1):
        root = pivot("Foot", (side * 0.100, -0.452, 0))
        root.rotation_euler.z = side * 0.12
        mb = MeshBuilder()
        mb.loft([(0, 0.004, z) for z in (0.20, 0.13, 0.085, 0.060)],
                [0.034, 0.029, 0.026, 0.028], ring=20, caps=(0.5, 0.6))
        shoe = ((0.022, 0.035, 0.030, 0.034), (-0.020, 0.038, 0.033, 0.037),
                (-0.080, 0.032, 0.028, 0.040), (-0.135, 0.026, 0.022, 0.038),
                (-0.168, 0.022, 0.016, 0.030))
        mb.loft([(0, y, z) for y, z, _, _ in shoe], [(ra, rb) for *_, ra, rb in shoe],
                ring=24, ref=(0, 0, 1), mats=1, caps=(0.7, 0.9))
        mb.blob((0, -0.072, 0.006), (0.041, 0.106, 0.006), exp=5, mat=1)
        mb.blob((0, -0.055, 0.058), (0.027, 0.046, 0.016), rot=(-0.35, 0, 0))
        mb.build("Ballet flat", [skin, leather]).parent = root


def build_head(head, skin, white, iris, hair, lip, frame_mat):
    """Head, face and hair cap, modelled in the head pivot's space (face toward -Y)."""
    C = Vector((0, -0.006, 0.050))
    half = (0.083, 0.097, 0.114)

    def warp(nv, p):
        x, y, z = p
        if nv.z < 0:                                       # narrower jaw
            k = (-nv.z) ** 1.4
            x *= 1 - 0.30 * k
            if nv.y > 0:
                y *= 1 - 0.30 * k
        if nv.y < 0:                                       # flatter face, fuller mouth and chin
            y *= 1 - 0.06 * smoothstep(0.6, 1.0, -nv.y)
            y *= 1 + 0.22 * smoothstep(0.0, 0.8, -nv.z) * smoothstep(0.3, 0.9, -nv.y)
            y *= 1 + 0.25 * smoothstep(0.75, 0.97, -nv.z) * smoothstep(0.0, 0.5, -nv.y)
        else:
            y *= 1.04                                      # rounder back of the skull
        return Vector((x, y, z))

    SKIN, WHITE, IRIS, LIP = range(4)
    mb = MeshBuilder()
    mb.blob(C, half, seg=40, rings=24, warp=warp)
    mb.loft([(0, 0.036, 0.0), (0, 0.040, -0.060), (0, 0.040, -0.125)],
            [0.043, 0.044, 0.046], ring=24, caps=(0.6, 0.6))
    for s in (-1, 1):
        x = s * 0.033
        mb.blob((x, -0.081, 0.045), (0.0125,) * 3, seg=20, rings=12, mat=WHITE)
        mb.blob((x, -0.0925, 0.045), (0.0068, 0.0028, 0.0068), seg=16, rings=8, mat=IRIS)
        mb.blob((x, -0.087, 0.057), (0.0158, 0.0105, 0.0072), seg=16, rings=8, rot=(-0.25, 0, 0))
        mb.blob((x, -0.086, 0.034), (0.0140, 0.0080, 0.0040), seg=16, rings=8)
        mb.blob((s * 0.082, 0.010, 0.040), (0.008, 0.016, 0.023), seg=16, rings=8,
                rot=(0, 0, -s * 0.3))
        mb.blob((s * 0.0095, -0.1015, 0.001), (0.0065, 0.0060, 0.0050), seg=12, rings=6)
        line("Eyelashes", [(x - 0.016, -0.0905, 0.048), (x, -0.0985, 0.0505),
                           (x + 0.016, -0.0905, 0.049), (x + s * 0.021, -0.088, 0.053)],
             0.0012, hair, head)
        line("Eyebrow", [(s * 0.013, -0.0975, 0.070), (s * 0.033, -0.1000, 0.077),
                         (s * 0.054, -0.0900, 0.073)], 0.0024, hair, head)
        rim = [(x + 0.0235 * sgnpow(math.cos(a), 0.5), -0.110 + s * 0.006 * math.cos(a),
                0.046 + 0.0165 * sgnpow(math.sin(a), 0.5))
               for a in (i * math.tau / 16 for i in range(16))]
        line("Eyeglass rim", rim, 0.0024, frame_mat, head, closed=True)
        line("Eyeglass temple", [(s * 0.057, -0.105, 0.052), (s * 0.085, -0.082, 0.054),
                                 (s * 0.091, -0.010, 0.050), (s * 0.088, 0.022, 0.034)],
             0.0021, frame_mat, head)
    line("Eyeglass bridge", [(-0.011, -0.111, 0.051), (0, -0.114, 0.055),
                            (0.011, -0.111, 0.051)], 0.0022, frame_mat, head)
    mb.loft([(0, -0.090, 0.052), (0, -0.099, 0.030), (0, -0.108, 0.010)],
            [0.0065, 0.0085, 0.0100], ring=16, caps=(0.5, 0.0))
    mb.blob((0, -0.1075, 0.004), (0.0105, 0.0095, 0.0085), seg=16, rings=8)
    mb.blob((0, -0.088, -0.022), (0.0170, 0.0065, 0.0048), seg=16, rings=8, mat=LIP,
            rot=(0.2, 0, 0))
    mb.blob((0, -0.087, -0.031), (0.0145, 0.0072, 0.0052), seg=16, rings=8, mat=LIP,
            rot=(-0.2, 0, 0))
    mb.build("Head", [skin, white, iris, lip]).parent = head

    def keep(nv):
        face = smoothstep(0.05, 0.45, -nv.y) * (1 - smoothstep(0.85, 0.98, abs(nv.x)))
        under_jaw = -0.45 - 0.65 * smoothstep(0.1, 0.6, nv.y)   # only the back reaches the nape
        return nv.z > max(-1.1 + 1.45 * face, under_jaw)

    def hair_warp(nv, p):
        x, y, z = p
        if nv.z < 0:                                       # hair falls straight past the jaw
            x *= 1 + 0.28 * smoothstep(0.0, 1.0, -nv.z)
        y *= 1.04 if nv.y > 0 else 1 - 0.06 * smoothstep(0.6, 1.0, -nv.y)
        return Vector((x, y, z))

    cap = MeshBuilder()
    cap.blob(C + Vector((0, 0.002, 0.004)), (half[0] * 1.07, half[1] * 1.06, half[2] * 1.06),
             seg=40, rings=24, warp=hair_warp, keep=keep)
    for pts in (((0.030, -0.060, 0.162), (0.000, -0.098, 0.134), (-0.040, -0.104, 0.108),
                 (-0.075, -0.086, 0.080), (-0.091, -0.050, 0.040)),
                ((0.030, -0.060, 0.162), (0.060, -0.090, 0.124), (0.084, -0.072, 0.082),
                 (0.093, -0.036, 0.040))):
        refs = [Vector(p) - C for p in pts]
        cap.loft(pts, [(0.006, 0.030), (0.007, 0.034), (0.007, 0.030), (0.006, 0.022),
                       (0.004, 0.012)][:len(pts)], ring=16, refs=refs)
    cap.build("Hair cap and fringe", [hair]).parent = head


def build_long_hair(chest, hair):
    mb = MeshBuilder()
    back = ((0.050, 1.250, 0.040, 0.080), (0.085, 1.150, 0.036, 0.100),
            (0.112, 1.060, 0.030, 0.122), (0.132, 0.960, 0.025, 0.128),
            (0.136, 0.880, 0.020, 0.118), (0.132, 0.820, 0.010, 0.085))
    mb.loft([(0, y, z) for y, z, _, _ in back], [(a, b) for *_, a, b in back], ring=28,
            ref=(0, 1, 0), caps=(0.5, 0.8))
    for side in (-1, 1):
        pts = [(side * x, y, z) for x, y, z in (
            (0.074, 0.000, 1.260), (0.090, -0.012, 1.190), (0.104, -0.022, 1.120),
            (0.124, -0.036, 1.060), (0.138, -0.056, 1.005), (0.144, -0.066, 0.965))]
        radii = [(0.014, 0.040), (0.015, 0.046), (0.015, 0.046), (0.014, 0.043),
                 (0.011, 0.036), (0.005, 0.022)]
        refs = [(1, 0, 0), (1, 0, 0), (1, 0.5, 0), (0.5, 1, 0), (0.1, 1, 0), (0, 1, 0)]
        mb.loft(pts, radii, ring=20, refs=refs, caps=(0.5, 0.9))
    attach(mb.build("Long hair", [hair]), chest, CHEST)


def build_arms(skin, shirt, trim):
    """Skinned arms; each bone follows an animated control empty."""
    data = bpy.data.armatures.new("Arm rig")
    rig = bpy.data.objects.new("Arm rig", data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    joints = {}
    for side, tag in ((-1, "R"), (1, "L")):
        S = shoulder_rest(side)
        E = S - Vector((0, 0, UPPER_ARM))
        W = E - Vector((0, 0, FOREARM))
        joints[side] = (tag, S, E)
        for name, head, tail in ((f"Upper.{tag}", S, E), (f"Fore.{tag}", E, W)):
            bone = data.edit_bones.new(name)
            bone.head, bone.tail = head, tail
    bpy.ops.object.mode_set(mode="OBJECT")

    controls = {}
    SKIN, SHIRT, TRIM = range(3)
    mb = MeshBuilder()
    for side, (tag, S, E) in joints.items():
        for part, rest in (("Upper", S), ("Fore", E)):
            target = pivot(f"{part} arm control.{tag}", rest)
            target.rotation_mode = "QUATERNION"
            con = rig.pose.bones[f"{part}.{tag}"].constraints.new("CHILD_OF")
            con.target = target
            con.inverse_matrix = Matrix.Translation(rest).inverted()
            controls[side, part] = target
        ss = stations(0.0, ARM_KEYS[-1][0], 0.010, [k[0] for k in ARM_KEYS])

        def weight(i, _, tag=tag, ss=ss):
            k = smoothstep(UPPER_ARM - 0.035, UPPER_ARM + 0.035, ss[i])
            return {f"Upper.{tag}": 1 - k, f"Fore.{tag}": k}

        mats = [SHIRT if s < 0.155 else TRIM if s < 0.168 else SKIN
                for s in ((a + b) / 2 for a, b in zip(ss, ss[1:]))]
        mb.loft([S - Vector((0, 0, s)) for s in ss], [profile_at(ARM_KEYS, s)[0] for s in ss],
                ring=24, mats=mats, caps=(0.9, 0.4), weight=weight)
    arms = mb.build("Arms", [skin, shirt, trim], subsurf=None)
    mod = arms.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    mod.use_deform_preserve_volume = True
    sub = arms.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return controls


def hand_axis(side):
    """Where the gripped tube runs through the hand, in the hand's space."""
    return Vector((-side * GRIP, -0.022, 0))


def hand_wrist(side):
    return Vector((side * 0.004, 0.050, -0.004))


def build_hand(side, skin):
    """Power grip around a tube along +Z; the forearm leaves the hand toward +Y."""
    hand = pivot("Right hand" if side < 0 else "Left hand")
    hand.rotation_mode = "QUATERNION"
    cx, cy, _ = hand_axis(side)
    mb = MeshBuilder()
    mb.blob((side * 0.002, 0.008, 0), (0.013, 0.044, 0.035), exp=2.4, seg=24, rings=12)
    mb.blob((side * 0.004, 0.030, 0.024), (0.013, 0.020, 0.015), seg=16, rings=10)
    for z, length, r in ((0.022, 0.074, 0.0085), (0.007, 0.080, 0.0088),
                         (-0.008, 0.075, 0.0085), (-0.022, 0.060, 0.0076)):
        knuckle = Vector((side * 0.004, -0.036, z))
        a0 = math.atan2(knuckle.y - cy, knuckle.x - cx)
        sweep = length / 0.027
        finger(mb, [knuckle] + [(cx + 0.027 * math.cos(a0 - side * sweep * k / 4),
                                 cy + 0.027 * math.sin(a0 - side * sweep * k / 4), z)
                                for k in range(1, 5)], r)
    start = Vector((side * 0.006, 0.026, 0.028))
    a0 = math.atan2(start.y - cy, start.x - cx)
    finger(mb, [start] + [(cx + 0.030 * math.cos(a0 + side * 0.45 * k),
                           cy + 0.030 * math.sin(a0 + side * 0.45 * k), 0.028 + 0.002 * k)
                          for k in range(1, 5)], 0.0105, taper=0.75)
    mb.build("Hand", [skin]).parent = hand
    return hand


# ---------------------------------------------------------------- props
def build_case(shell, lining, trim):
    root = pivot("Open case on lap", (0, -0.318, 0.644))
    box("Case base", (0, 0, -0.022), (0.646, 0.272, 0.037), shell, parent=root)
    box("Velvet seat", (0, 0, -0.008), (0.614, 0.239, 0.020), lining, parent=root)
    for side in (-1, 1):
        box("Case long rim", (0, side * 0.126, 0.004), (0.644, 0.020, 0.046),
            shell, parent=root)
        box("Case end rim", (side * 0.312, 0, 0.004), (0.023, 0.247, 0.046),
            shell, parent=root)
    for y in (-0.084, -0.027, 0.027, 0.082):
        box("Padded slot divider", (0, y, 0.001), (0.578, 0.010, 0.010),
            lining, bevel=0.004, parent=root)
    lid = pivot("Fixed open lid", (0, -0.130, 0.013), root)
    lid.rotation_euler.x = math.radians(205)
    box("Case lid", (0, 0.130, 0), (0.646, 0.266, 0.031), shell, parent=lid)
    box("Lid velvet", (0, 0.130, -0.018), (0.615, 0.236, 0.013), lining, parent=lid)
    for side in (-1, 1):
        box("Latch", (side * 0.207, 0.263, 0.003), (0.036, 0.012, 0.023),
            trim, bevel=0.003, parent=lid)
    line("Carry handle", [(-0.074, 0.265, 0), (-0.053, 0.294, 0),
                          (0.053, 0.294, 0), (0.074, 0.265, 0)],
         0.009, shell, lid)


def build_instrument(wood, silver, reed, cork):
    names = ("Bell", "Lower joint", "Upper joint", "Barrel and mouthpiece")
    lengths = (0.112, 0.207, 0.193, 0.137)
    pieces = []
    for i, (name, length) in enumerate(zip(names, lengths)):
        part = pivot(name)
        if i == 0:
            rings = ((0.052, 0.012), (0.001, 0.044), (0, 0.050),
                     (0.004, 0.052), (0.015, 0.049), (0.042, 0.031),
                     (0.083, 0.019), (length, 0.016), (length, 0.012), (0.052, 0.012))
        elif i == 3:
            rings = ((0, 0.012), (0, 0.017), (0.017, 0.020), (0.052, 0.019),
                     (0.063, 0.015), (0.101, 0.014), (length, 0.009), (length, 0.001))
        else:
            rings = ((0, 0.011), (0, 0.016), (0.013, 0.018), (length - 0.01, 0.016),
                     (length, 0.015), (length, 0.011))
        profile(f"{name} body", rings, wood, part)
        for z, radius in ((0.007, 0.051 if i == 0 else 0.019),
                          (0.040 if i == 3 else length - 0.009, 0.017)):
            profile("Silver collar", ((z - 0.002, radius - 0.001), (z, radius),
                                     (z + 0.003, radius), (z + 0.004, radius - 0.001)),
                    silver, part)
        if i:
            profile("Cork tenon", ((-0.017, 0.009), (-0.016, 0.013), (0, 0.013)),
                    cork, part)
        if i in (1, 2):
            line("Key rod", [(0.016, -0.005, 0.024), (0.017, -0.005, length - 0.019)],
                 0.0027, silver, part)
            for j in range(5):
                z = 0.034 + j * 0.029
                x = (-1) ** j * 0.009
                oval("Silver key", (x, -0.016, z), (0.010, 0.005, 0.009), silver, part)
                oval("Key center", (x, -0.020, z), (0.004, 0.001, 0.004), wood, part)
                line("Key lever", [(x, -0.018, z), (0.015, -0.011, z + 0.009)],
                     0.0020, silver, part)
        if i == 3:
            oval("Reed", (0, -0.012, 0.108), (0.009, 0.004, 0.028), reed, part)
        pieces.append(part)
    return pieces, lengths


# ---------------------------------------------------------------- animation
def ease(t, start, end):
    u = max(0.0, min(1.0, (t - start) / (end - start)))
    return u * u * (3 - 2 * u)


def key(obj, frame, *properties):
    for prop in properties:
        obj.keyframe_insert(prop, frame=frame)


def grip_rotation(axis, toward):
    """Hand orientation: +Z along the gripped tube, +Y (the forearm) toward `toward`."""
    z = axis.normalized()
    y = toward - z * toward.dot(z)
    if y.length < 1e-6:
        y = Vector((0, 0, 1)) - z * z.z
    y.normalize()
    return Matrix((y.cross(z), y, z)).transposed().to_quaternion()


def shoulder(side, lean):
    return CHEST + Quaternion((1, 0, 0), lean) @ (shoulder_rest(side) - CHEST)


def wrist(side, grip, rotation):
    return grip + rotation @ (hand_wrist(side) - hand_axis(side))


def solve_elbow(S, W, pole):
    """Two-bone IK with the elbow bending toward `pole`; returns the elbow and any shortfall."""
    d = W - S
    u = d.normalized()
    dist = min(d.length, (UPPER_ARM + FOREARM) * 0.999)
    cos_a = (UPPER_ARM ** 2 + dist ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * dist)
    cos_a = max(-1.0, min(1.0, cos_a))
    p = (pole - u * pole.dot(u)).normalized()
    return S + (u * cos_a + p * math.sqrt(1 - cos_a ** 2)) * UPPER_ARM, d.length - dist


def segment_rotation(top, bottom, bend_normal):
    z = (top - bottom).normalized()
    x = -bend_normal
    x = (x - z * x.dot(z)).normalized()
    return Matrix((x, z.cross(x), z)).transposed().to_quaternion()


def animate(chest, head, pieces, lengths, hands, controls):
    tray_rotation = Quaternion((0, 1, 0), math.pi / 2)
    tray = [Vector(p) for p in ((-0.249, -0.314, 0.707), (-0.056, -0.405, 0.676),
                                (0.052, -0.348, 0.676), (0.086, -0.233, 0.677))]
    # The bell stays in one hand; the other brings each tenon to the exposed socket.
    stages = ((1.0, 2.45, 4.2, 4.2, 4.2, 4.2),
              (4.4, 5.45, 6.7, 7.8, 8.45, 9.0),
              (9.1, 10.1, 11.35, 12.45, 13.10, 13.65),
              (13.75, 14.75, 15.85, 16.9, 17.50, 18.0))
    grips = (0.074, 0.100, 0.095, 0.045)
    rest = {s: Vector((s * 0.300, -0.300, 0.700)) for s in (-1, 1)}
    rest_rot = {s: grip_rotation(Vector((0, -1, 0)), shoulder_rest(s) - rest[s]) for s in (-1, 1)}

    timeline = []
    for frame in range(1, FRAMES + 1):
        t = (frame - 1) / FPS
        present = ease(t, 18.0, 19.1)
        base = Vector((-0.115, -0.400, 0.735)).lerp(Vector((-0.090, -0.420, 0.760)), present)
        assembled = Quaternion((0, 1, 0), 0.62 - 0.10 * present) @ Quaternion((1, 0, 0), 0.10)
        axis = assembled @ Vector((0, 0, 1))
        held = {-1: None, 1: None}
        for i, part in enumerate(pieces):
            reach, grab, clear, align, seat, release = stages[i]
            socket = base + axis * sum(lengths[:i])
            if i == 0:
                lift = ease(t, grab, clear)
                location = tray[0].lerp(base, lift) + Vector((0, -0.04, 0.10)) * math.sin(math.pi * lift)
                rotation = tray_rotation.slerp(assembled, lift)
            else:
                lifted = tray[i] + Vector((0.02, -0.04, 0.17))
                approach = socket + axis * 0.048
                location = (tray[i].lerp(lifted, ease(t, grab, clear))
                            if t < clear else lifted.lerp(approach, ease(t, clear, align))
                            if t < align else approach.lerp(socket, ease(t, align, seat)))
                rotation = tray_rotation.slerp(assembled, ease(t, grab, align))
                rotation = rotation @ Quaternion((0, 0, 1), 0.36 * (1 - ease(t, align, seat))
                                                 * ease(t, clear, align))
            part.location = location
            part.rotation_mode = "QUATERNION"
            part.rotation_quaternion = rotation
            key(part, frame, "location", "rotation_quaternion")
            if t >= reach and (i == 0 or t <= release):
                hold = ease(t, reach, grab)
                if i:
                    hold *= 1 - ease(t, seat + 0.12, release)
                held[-1 if i == 0 else 1] = (location + rotation @ Vector((0, 0, grips[i])),
                                             rotation @ Vector((0, 0, 1)), hold)
        if t >= 18.0:
            held[1] = (base + axis * 0.40, axis, present)
        targets = {}
        for s in (-1, 1):
            if held[s] is None:
                targets[s] = (rest[s], rest_rot[s])
            else:
                grip, tube, w = held[s]
                targets[s] = (rest[s].lerp(grip, w),
                              rest_rot[s].slerp(grip_rotation(tube, shoulder_rest(s) - grip), w))
        timeline.append((t, present, targets))

    # Lean just far enough for both wrists to stay within reach, smoothed over ~0.8 s.
    def needed(targets):
        for k in range(61):
            lean = k * 0.005
            if all((wrist(s, *targets[s]) - shoulder(s, lean)).length <= REACH for s in (-1, 1)):
                return lean
        return 0.30

    need = [needed(targets) for _, _, targets in timeline]
    r = 12
    peak = [max(need[max(0, i - r):i + r + 1]) for i in range(len(need))]
    leans = [0.04 + sum(peak[max(0, i - r):i + r + 1]) / len(peak[max(0, i - r):i + r + 1])
             for i in range(len(peak))]

    worst = 0.0
    normals = {}
    for frame, ((t, present, targets), lean) in enumerate(zip(timeline, leans), start=1):
        chest.rotation_euler.x = lean
        key(chest, frame, "rotation_euler")
        head.rotation_euler.x = 0.24 + 0.10 * math.sin(math.pi * ease(t, 1, 18)) - 0.24 * present - lean
        head.rotation_euler.z = -0.07 + 0.18 * ease(t, 4.1, 5.5) - 0.10 * present
        key(head, frame, "rotation_euler")
        tilt = Quaternion((1, 0, 0), lean)
        for s in (-1, 1):
            grip, rotation = targets[s]
            hands[s].location = grip - rotation @ hand_axis(s)
            hands[s].rotation_quaternion = rotation
            key(hands[s], frame, "location", "rotation_quaternion")
            S, W = shoulder(s, lean), wrist(s, grip, rotation)
            E, gap = solve_elbow(S, W, tilt @ Vector((s * 0.55, 0.45, -1.0)))
            worst = max(worst, gap)
            W = E + (W - E).normalized() * FOREARM
            n = (E - S).cross(W - E)
            normals[s] = n.normalized() if n.length > 1e-5 else normals.get(s, Vector((-1, 0, 0)))
            for part, location, rot in (("Upper", S, segment_rotation(S, E, normals[s])),
                                        ("Fore", E, segment_rotation(E, W, normals[s]))):
                control = controls[s, part]
                control.location = location
                control.rotation_quaternion = rot
                key(control, frame, "location", "rotation_quaternion")
    print(f"Arm reach: lean {min(leans):.3f}-{max(leans):.3f} rad, largest wrist gap {worst * 1000:.1f} mm")


def studio(scene, wood, fabric, floor):
    box("Studio floor", (0, 0, -0.055), (200, 200, 0.1), floor, bevel=0.005)
    oval("Oval rug", (0, -0.079, 0.002), (0.87, 0.73, 0.012), fabric)
    box("Chair cushion", (0, 0.012, 0.469), (0.470, 0.415, 0.094), fabric, bevel=0.038)
    box("Chair seat frame", (0, 0.018, 0.414), (0.466, 0.414, 0.048), wood)
    for side in (-1, 1):
        for y in (-0.155, 0.182):
            leg = box("Chair leg", (side * 0.193, y, 0.213), (0.043, 0.048, 0.41), wood)
            leg.rotation_euler.y = side * -0.078
        box("Chair back post", (side * 0.199, 0.202, 0.685), (0.039, 0.038, 0.51), wood)
    box("Chair backrest", (0, 0.207, 0.905), (0.450, 0.063, 0.112), wood, bevel=0.028)
    scene.world = bpy.data.worlds.new("Studio ambience")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.28, 0.34, 0.42, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    for name, pos, energy, size, color in (
        ("Warm softbox", (-2.3, -3.2, 3.9), 440, 3.0, (1.0, 0.86, 0.72)),
        ("Cool fill", (2.5, -0.9, 2.6), 240, 2.8, (0.72, 0.86, 1.0)),
        ("Hair light", (-0.2, 2.2, 3.2), 415, 2.1, (1.0, 0.76, 0.49)),
    ):
        bpy.ops.object.light_add(type="AREA", location=pos)
        lamp = bpy.context.object
        lamp.name = name
        lamp.data.energy, lamp.data.size, lamp.data.color = energy, size, color
        lamp.rotation_euler = (Vector((0, -0.12, 0.8)) - lamp.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 48
    scene.camera = camera
    for frame, position, target in (
        (1, (1.49, -3.64, 2.18), (0, -0.19, 0.69)),
        (125, (1.43, -3.26, 2.06), (0.01, -0.28, 0.81)),
        (490, (1.39, -3.18, 1.97), (0.01, -0.29, 0.82)),
        (FRAMES, (1.39, -3.57, 2.04), (0, -0.21, 0.70)),
    ):
        camera.location = position
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
        key(camera, frame, "location", "rotation_euler")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--still", type=int, nargs="+", default=[])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--preview", action="store_true")
    opts = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if any(not 1 <= frame <= FRAMES for frame in opts.still):
        parser.error(f"--still frames must be between 1 and {FRAMES}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    skin = material("Warm skin", (0.60, 0.31, 0.205), 0.45)
    shirt = material("Deep teal cotton", (0.018, 0.30, 0.275), 0.7, bump=0.08)
    shirt_trim = material("Teal piping", (0.010, 0.19, 0.175), 0.6)
    skirt = material("Black skirt", (0.015, 0.016, 0.021), 0.87, bump=0.06)
    leather = material("Black leather flats", (0.012, 0.012, 0.014), 0.33)
    hair = hair_material()
    frame_mat = material("Dark tortoiseshell", (0.090, 0.036, 0.016), 0.32)
    white = material("Eye white", (0.88, 0.81, 0.71), 0.25)
    iris = material("Brown iris", (0.035, 0.018, 0.010), 0.15)
    lip = material("Muted rose", (0.32, 0.083, 0.065), 0.4)
    blackwood = material("Polished grenadilla", (0.021, 0.017, 0.016), 0.27)
    silver = material("Nickel silver", (0.66, 0.72, 0.78), 0.28, 0.77)
    reed = material("Cane reed", (0.64, 0.41, 0.16))
    cork = material("Joint cork", (0.48, 0.29, 0.14))
    shell = material("Case charcoal", (0.026, 0.039, 0.044))
    lining = material("Wine velvet", (0.207, 0.042, 0.069), 0.92)
    oak = material("Smoked oak", (0.26, 0.13, 0.061))
    fabric = material("Sand fabric", (0.43, 0.30, 0.19), 0.86)
    floor = material("Muted sage studio", (0.23, 0.32, 0.287), 0.92)
    studio(scene, oak, fabric, floor)

    chest = pivot("Upper body", CHEST)
    head = pivot("Head", HEAD_PIVOT - CHEST, chest)
    build_upper_body(chest, skin, shirt, shirt_trim)
    build_skirt(skirt)
    build_feet(skin, leather)
    build_head(head, skin, white, iris, hair, lip, frame_mat)
    build_long_hair(chest, hair)
    controls = build_arms(skin, shirt, shirt_trim)
    hands = {side: build_hand(side, skin) for side in (-1, 1)}
    build_case(shell, lining, silver)
    pieces, lengths = build_instrument(blackwood, silver, reed, cork)
    animate(chest, head, pieces, lengths, hands, controls)

    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.render.image_settings.file_format = "PNG"
    out = HERE / "render"
    scene.render.filepath = str(out / "clarinet_assembly")
    scene.frame_set(1)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE / "clarinet_assembly.blend"))
    if opts.preview:
        scene.eevee.taa_render_samples = 8
        scene.render.resolution_percentage = 50
    if opts.still or opts.render:
        out.mkdir(exist_ok=True)
    for frame in opts.still:
        scene.frame_set(frame)
        scene.render.filepath = str(out / f"clarinet_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)
    if opts.render:
        settings = scene.render.image_settings
        if hasattr(settings, "media_type"):
            settings.media_type = "VIDEO"
        settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.filepath = str(out / "clarinet_assembly")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
