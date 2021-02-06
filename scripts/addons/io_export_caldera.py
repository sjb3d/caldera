bl_info = {
    "name": "Caldera Export",
    "blender": (2, 91, 0),
    "category": "Import-Export",
}

'''
Mesh "id"
{
    %f %f %f
    %f %f %f
    ...
}
{
    %u %u %u
    %u %u %u
    ...
}

Transform "id"
%f %f %f %f
%f %f %f %f
%f %f %f %f

Instance "transform_id" "geometry_id"

Camera "transform_id" %f
'''

import bpy
import os
import math
import mathutils
from bpy_extras.io_utils import ExportHelper

class Mesh:
    def __init__(self, mesh, bake_scale):
        self.positions = list()
        self.indices = list()
        for v in mesh.vertices:
            if bake_scale:
                self.positions.append((v.co * bake_scale)[:])
            else:
                self.positions.append(v.co[:])
        for p in mesh.polygons:
            for idx in range(2, len(p.vertices)):
                tri = (p.vertices[0], p.vertices[idx - 1], p.vertices[idx])
                self.indices.append(tri)

    def matches(self, other):
        if len(self.positions) != len(other.positions):
            return False
        if len(self.indices) != len(other.indices):
            return False
        
        for (a, b) in zip(self.positions, other.positions):
            if a != b:
                return False
        for (a, b) in zip(self.indices, other.indices):
            if a != b:
                return False

        return True


class ExportCaldera(bpy.types.Operator, ExportHelper):
    '''Export scene in caldera trace app format'''
    bl_idname = "export.caldera"
    bl_label = "Export Caldera"
    filename_ext = ".caldera"

    meshes = list()
    transforms = list()

    def reset(self):
        self.meshes.clear()
        self.transforms.clear()

    def export_mesh_data(self, fw, ob, mesh, bake_scale):
        check = Mesh(mesh, bake_scale)
        if len(check.indices) == 0:
            return None

        for (key, other) in self.meshes:
            if check.matches(other):
                return key

        key = '%i-%s' % (len(self.meshes), ob.data.name)
        self.meshes.append((key, check))
    
        fw('mesh "%s"\n' % key)
        fw('{\n')
        for v in check.positions:
            fw('%f %f %f\n' % v)
        fw('}\n')
        fw('{\n')
        for t in check.indices:
            fw('%i %i %i\n' % t)
        fw('}\n')
        fw('\n')

        return key

    def export_transform(self, fw, name, m):
        (translation, rotation, nu_scale) = m.decompose()

        sign_match = lambda x, y : (x * y) > 0.0

        if sign_match(nu_scale.y, nu_scale.z) and not sign_match(nu_scale.x, nu_scale.y):
            rotation = rotation @ mathutils.Quaternion((1.0, 0.0, 0.0), math.pi)
            nu_scale.y = -nu_scale.y
            nu_scale.z = -nu_scale.z
        if sign_match(nu_scale.z, nu_scale.x) and not sign_match(nu_scale.y, nu_scale.z):
            rotation = rotation @ mathutils.Quaternion((0.0, 1.0, 0.0), math.pi)
            nu_scale.z = -nu_scale.z
            nu_scale.x = -nu_scale.x
        if sign_match(nu_scale.x, nu_scale.y) and not sign_match(nu_scale.z, nu_scale.x):
            rotation = rotation @ mathutils.Quaternion((0.0, 0.0, 1.0), math.pi)
            nu_scale.x = -nu_scale.x
            nu_scale.y = -nu_scale.y

        #check = mathutils.Matrix.Translation(translation) @ rotation.to_matrix().to_4x4() @ mathutils.Matrix.Diagonal(nu_scale).to_4x4()
        #print(m - check)

        scale = nu_scale.x
        nu_scale = nu_scale/scale

        bake_scale = None
        if abs(1.0 - nu_scale.y) > 0.01 or abs(1.0 - nu_scale.z) > 0.01:
            bake_scale = nu_scale

        for (key, other) in self.transforms:
            if (translation, rotation, scale) == other:
                return (key, bake_scale)

        key = '%i-%s' % (len(self.transforms), name)
        self.transforms.append((key, (translation, rotation, scale)))

        fw('transform "%s"\n' % key)
        fw('%f %f %f\n' % translation[:])
        fw('%f %f %f %f\n' % rotation[:])
        fw('%f\n' % scale)

        fw('\n')

        return (key, bake_scale)

    def translate_material(self, material):
        if not material:
            return None
        tree = material.node_tree
        if not tree:
            return None

        input_nodes = dict([[link.to_socket, link.from_node] for link in tree.links])
        for node in tree.nodes:
            if node.bl_idname == 'ShaderNodeOutputMaterial':
                surface_node = input_nodes[node.inputs['Surface']]
                if surface_node:
                    if surface_node.bl_idname == 'ShaderNodeBsdfGlossy':
                        if surface_node.distribution == 'BECKMANN':
                            return ("mirror", 0.5, 0.0, 0.0)
                    
        return ("diffuse",) + tuple(material.diffuse_color[:3])
                

    def export_mesh(self, fw, ob, m):
        mesh = ob.to_mesh()

        (transform_key, bake_scale) = self.export_transform(fw, ob.name, m)
        mesh_key = self.export_mesh_data(fw, ob, mesh, bake_scale)

        if mesh_key is not None:
            material = None
            if len(mesh.materials) > 0:
                material = self.translate_material(mesh.materials[0])
            if not material:
                material = ("diffuse", 0.8, 0.8, 0.8)

            fw('instance "%s" "%s"\n' % (transform_key, mesh_key))
            fw('%s %f %f %f' % material)
            fw('\n')

        ob.to_mesh_clear()

    def export_camera(self, fw, ob, m):
        # rotate to look down position z
        adjusted_m = m @ mathutils.Matrix.Rotation(math.pi, 4, 'Y')
        (transform_key, bake_scale) = self.export_transform(fw, ob.name, adjusted_m)

        fw('camera "%s" %f\n' % (transform_key, ob.data.angle_y))
        fw('\n')

    def execute(self, context):
        self.reset()

        file = open(self.filepath, "w")
        fw = file.write

        dg = context.evaluated_depsgraph_get()
        for dup in dg.object_instances:
            ob = dup.instance_object if dup.is_instance else dup.object
            m = mathutils.Matrix(dup.matrix_world)
            if ob.type == 'MESH':
                self.export_mesh(fw, ob, m)
            elif ob.type == 'CAMERA':
                self.export_camera(fw, ob, m)
  
        file.close()
        self.reset()
        return {'FINISHED'}

def menu_func_export(self, context):
    self.layout.operator(ExportCaldera.bl_idname, text="Caldera (.caldera)")

classes=[
    ExportCaldera,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for c in classes:
        bpy.utils.unregister_class(c)
