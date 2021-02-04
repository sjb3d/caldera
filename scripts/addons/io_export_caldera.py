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

    seen_meshes = set()

    def reset_seen(self):
        self.seen_meshes.clear()

    def export_mesh_data(self, fw, mesh):
        # only export once
        if mesh.name in self.seen_meshes:
            return
        self.seen_meshes.add(mesh.name)
        
        fw('mesh "%s"\n' % mesh.name)
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

    def export_transform(self, fw, ob, m):
        fw('transform "%s"\n' % ob.name)
        fw('%f %f %f %f\n' % m[0][:])
        fw('%f %f %f %f\n' % m[1][:])
        fw('%f %f %f %f\n' % m[2][:])
        fw('\n')

    def export_mesh(self, fw, ob, m):
        if len(ob.data.polygons) == 0:
            return

        self.export_transform(fw, ob, m)
        self.export_mesh_data(fw, ob.data)

        fw('instance "%s" "%s"\n' % (ob.name, ob.data.name))
        fw('\n')

    def export_camera(self, fw, ob, m):
        # rotate to look down position z
        adjusted_m = m @ mathutils.Matrix.Rotation(math.pi, 4, 'Y')
        self.export_transform(fw, ob, adjusted_m)

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
