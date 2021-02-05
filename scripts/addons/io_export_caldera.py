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

class ExportCaldera(bpy.types.Operator, ExportHelper):
    '''Export scene in caldera trace app format'''
    bl_idname = "export.caldera"
    bl_label = "Export Caldera"
    filename_ext = ".caldera"

    seen_meshes = dict()

    def reset_seen(self):
        self.seen_meshes.clear()

    def export_mesh_data(self, fw, ob):
        # only export once
        name = ob.data.name
        is_valid = self.seen_meshes.get(name)
        if is_valid is None:
            mesh = ob.to_mesh()
            is_valid = (len(mesh.polygons) > 0)
            if is_valid:
                fw('mesh "%s"\n' % name)
                fw('{\n')
                for v in mesh.vertices:
                    fw('%f %f %f\n' % v.co[:])
                fw('}\n')
                fw('{\n')
                for p in mesh.polygons:
                    for idx in range(2, len(p.vertices)):
                        fw('%i %i %i\n' % (p.vertices[0], p.vertices[idx - 1], p.vertices[idx]))
                fw('}\n')
                fw('\n')
            ob.to_mesh_clear()
            self.seen_meshes[name] = is_valid
        return name if is_valid else None

    def export_transform(self, fw, name, m):
        fw('transform "%s"\n' % name)
        fw('%f %f %f %f\n' % m[0][:])
        fw('%f %f %f %f\n' % m[1][:])
        fw('%f %f %f %f\n' % m[2][:])
        fw('\n')

    def export_mesh(self, fw, ob, m):
        mesh_name = self.export_mesh_data(fw, ob)
        if mesh_name is not None:
            self.export_transform(fw, ob.name, m)

            fw('instance "%s" "%s"\n' % (ob.name, mesh_name))
            fw('\n')

    def export_camera(self, fw, ob, m):
        # rotate to look down position z
        adjusted_m = m @ mathutils.Matrix.Rotation(math.pi, 4, 'Y')
        self.export_transform(fw, ob.name, adjusted_m)

        fw('camera "%s" %f\n' % (ob.name, ob.data.angle_y))
        fw('\n')

    def execute(self, context):
        self.reset_seen()

        file = open(self.filepath, "w")
        fw = file.write

        dg = context.evaluated_depsgraph_get()
        for object_instance in dg.object_instances:
            ob = object_instance.object
            if ob.type == 'MESH':
                self.export_mesh(fw, ob, object_instance.matrix_world)
            elif ob.type == 'CAMERA':
                self.export_camera(fw, ob, object_instance.matrix_world)
  
        file.close()
        self.reset_seen()
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
