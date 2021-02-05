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
    def __init__(self, mesh):
        self.positions = list()
        self.indices = list()
        for v in mesh.vertices:
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
    transform_count = 0

    def reset(self):
        self.meshes.clear()
        self.transform_count = 0

    def export_mesh_data(self, fw, ob):
        mesh = ob.to_mesh()
        check = Mesh(mesh)
        ob.to_mesh_clear()

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

        return key

    def export_transform(self, fw, name, m):
        key = '%i-%s' % (self.transform_count, name)
        self.transform_count += 1

        fw('transform "%s"\n' % key)
        fw('%f %f %f %f\n' % m[0][:])
        fw('%f %f %f %f\n' % m[1][:])
        fw('%f %f %f %f\n' % m[2][:])
        fw('\n')

        return key

    def export_mesh(self, fw, ob, m):
        mesh_key = self.export_mesh_data(fw, ob)
        if mesh_key is not None:
            transform_key = self.export_transform(fw, ob.name, m)

            fw('instance "%s" "%s"\n' % (transform_key, mesh_key))
            fw('\n')

    def export_camera(self, fw, ob, m):
        # rotate to look down position z
        adjusted_m = m @ mathutils.Matrix.Rotation(math.pi, 4, 'Y')
        transform_key = self.export_transform(fw, ob.name, adjusted_m)

        fw('camera "%s" %f\n' % (transform_key, ob.data.angle_y))
        fw('\n')

    def execute(self, context):
        self.reset()

        file = open(self.filepath, "w")
        fw = file.write

        dg = context.evaluated_depsgraph_get()
        for dup in dg.object_instances:
            ob = dup.instance_object if dup.is_instance else dup.object
            if ob.type == 'MESH':
                self.export_mesh(fw, ob, dup.matrix_world)
            elif ob.type == 'CAMERA':
                self.export_camera(fw, ob, dup.matrix_world)
  
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
