use crate::scene::*;
use caldera::*;
use std::mem;
use std::sync::Arc;

#[repr(C)]
struct PositionData([f32; 3]);

#[repr(C)]
struct IndexData([u32; 3]);

#[derive(Clone, Copy)]
struct TriangleMeshData {
    position_buffer: StaticBufferHandle,
    index_buffer: StaticBufferHandle,
}

struct BottomLevelAccel {
    geometries: Vec<GeometryRef>,
}

pub struct SceneAccel {
    scene: Arc<Scene>,
    geometry_data: Vec<TriangleMeshData>,
}

impl SceneAccel {
    pub fn new(scene: Scene, context: &Arc<Context>, resource_loader: &mut ResourceLoader) -> Self {
        let scene = Arc::new(scene);

        // make vertex/index buffers for each geometry
        let geometry_data = scene
            .geometry_ref_iter()
            .map(|geometry_ref| {
                let data = TriangleMeshData {
                    position_buffer: resource_loader.create_buffer(),
                    index_buffer: resource_loader.create_buffer(),
                };
                resource_loader.async_load({
                    let scene = Arc::clone(&scene);
                    move |allocator| {
                        let geometry = scene.geometry(geometry_ref).unwrap();
                        let position_buffer_desc =
                            BufferDesc::new(geometry.positions.len() * mem::size_of::<PositionData>());
                        let mut writer = allocator
                            .map_buffer(
                                data.position_buffer,
                                &position_buffer_desc,
                                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                            )
                            .unwrap();
                        for pos in geometry.positions.iter() {
                            writer.write_all(pos.as_byte_slice());
                        }

                        let index_buffer_desc = BufferDesc::new(geometry.indices.len() * mem::size_of::<IndexData>());
                        let mut writer = allocator
                            .map_buffer(
                                data.index_buffer,
                                &index_buffer_desc,
                                BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT | BufferUsage::RAY_TRACING_STORAGE_READ,
                            )
                            .unwrap();
                        for face in geometry.indices.iter() {
                            writer.write_all(face.as_byte_slice());
                        }
                    }
                });
                data
            })
            .collect();

        Self { scene, geometry_data }
    }

    pub fn update(&mut self) {
        
    }
}
