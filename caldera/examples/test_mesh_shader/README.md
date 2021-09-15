# test_mesh_shader

Run using:

```
make && cargo run --release --example test_mesh_shader -- <ply_file_name>
```

Example when run with the dragon model from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/) as `<ply_file_name>`.

![test_mesh_shader image](../../../images/test_mesh_shader.jpg)

## Clustering

The clustering code is extremely minimal: just grows clusters using adjacent triangles until the maximum number of vertices of triangles is reached.  If there is no adjacent triangle then the next triangle in the mesh is added.  A better clustering algorithm would score candidate triangles according to how they affect the position and normal bounds of the cluster.

## Backface Culling

To draw the mesh using clusters, the `VK_NV_mesh_shader` extension is required.  This makes use of:
- A task shader to backface cull clusters that fully face away from the viewer
- A mesh shader that emits all triangles in the cluster for rasterization (no addition culling)

The task shader makes use of `GL_KHR_shader_subgroup_ballot` to efficiently emit only the surviving clusters to the mesh stage. To achieve this, the task shader runs as a single subgroup, which seemed to require the following steps:

- When writing the GLSL code for the task shader:
  - Add a specialization constant to represent the subgroup size
  - Use `layout(local_size_x_id = <id>)` to make the workgroup size defined by the constant
  - Write the task shader with a single `subgroupBallotExclusiveBitCount` to pack outputs
- At application start:
  - Load the `VK_EXT_subgroup_size_control` extension
  - Check `VkPhysicalDeviceSubgroupSizeControlPropertiesEXT` for the subgroup size ranges
- When specifying the task shader stage for the pipeline:
  - Set the specialization constant to the desired subgroup size
  - Specify the subgroup size using `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`
  - Additionally specify a `VkPipelineShaderStageCreateFlags` of `REQUIRE_FULL_SUBGROUPS_EXT`

This results in a subgroup size of 32 (as expected) on a RTX 2060.
