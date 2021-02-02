use caldera::*;
use ply_rs::{parser, ply};
use std::{fs, io, mem};

#[derive(Clone, Copy)]
struct PlyVertex {
    pos: Vec3,
}

#[derive(Clone, Copy)]
struct PlyFace {
    indices: UVec3,
}

impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        Self { pos: Vec3::zero() }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.pos.x = v,
            ("y", ply::Property::Float(v)) => self.pos.y = v,
            ("z", ply::Property::Float(v)) => self.pos.z = v,
            _ => {}
        }
    }
}

impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        Self { indices: UVec3::zero() }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListInt(v)) => {
                assert_eq!(v.len(), 3);
                for (dst, src) in self.indices.as_mut_slice().iter_mut().zip(v.iter().rev()) {
                    *dst = *src as u32;
                }
            }
            (k, _) => panic!("unknown key {}", k),
        }
    }
}

pub type PositionData = Vec3;
pub type AttributeData = Vec3;
pub type InstanceData = TransposedTransform3;

#[derive(Clone, Copy)]
pub struct MeshInfo {
    pub position_buffer: StaticBufferHandle,
    pub attribute_buffer: StaticBufferHandle,
    pub index_buffer: StaticBufferHandle,
    pub instances: [Similarity3; Self::INSTANCE_COUNT],
    pub instance_buffer: StaticBufferHandle,
    pub vertex_count: u32,
    pub triangle_count: u32,
}

impl MeshInfo {
    pub const INSTANCE_COUNT: usize = 8;

    pub fn new(resource_loader: &mut ResourceLoader) -> Self {
        Self {
            position_buffer: resource_loader.create_buffer(),
            attribute_buffer: resource_loader.create_buffer(),
            index_buffer: resource_loader.create_buffer(),
            instances: [Similarity3::identity(); Self::INSTANCE_COUNT],
            instance_buffer: resource_loader.create_buffer(),
            vertex_count: 0,
            triangle_count: 0,
        }
    }

    pub fn load(&mut self, allocator: &mut ResourceAllocator, mesh_file_name: &str, with_ray_tracing: bool) {
        let vertex_parser = parser::Parser::<PlyVertex>::new();
        let face_parser = parser::Parser::<PlyFace>::new();

        let mut f = io::BufReader::new(fs::File::open(mesh_file_name).unwrap());
        let header = vertex_parser.read_header(&mut f).unwrap();

        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for (_key, element) in header.elements.iter() {
            match element.name.as_ref() {
                "vertex" => {
                    vertices = vertex_parser
                        .read_payload_for_element(&mut f, element, &header)
                        .unwrap();
                }
                "face" => {
                    faces = face_parser.read_payload_for_element(&mut f, element, &header).unwrap();
                }
                _ => panic!("unexpected element {:?}", element),
            }
        }

        let position_buffer_desc = BufferDesc::new(vertices.len() * mem::size_of::<PositionData>());
        let mut writer = allocator
            .map_buffer(
                self.position_buffer,
                &position_buffer_desc,
                if with_ray_tracing {
                    BufferUsage::VERTEX_BUFFER | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
                } else {
                    BufferUsage::VERTEX_BUFFER
                },
            )
            .unwrap();
        let mut min = Vec3::broadcast(f32::MAX);
        let mut max = Vec3::broadcast(-f32::MAX);
        for src in vertices.iter() {
            let v = src.pos;
            writer.write(&v);
            min = min.min_by_component(v);
            max = max.max_by_component(v);
        }
        self.vertex_count = vertices.len() as u32;

        let index_buffer_desc = BufferDesc::new(faces.len() * 3 * mem::size_of::<u32>());
        let mut writer = allocator
            .map_buffer(
                self.index_buffer,
                &index_buffer_desc,
                if with_ray_tracing {
                    BufferUsage::INDEX_BUFFER | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT
                } else {
                    BufferUsage::INDEX_BUFFER
                },
            )
            .unwrap();
        let mut normals = vec![Vec3::zero(); vertices.len()];
        for src in faces.iter() {
            writer.write(&src.indices);
            let v0 = vertices[src.indices[0] as usize].pos;
            let v1 = vertices[src.indices[1] as usize].pos;
            let v2 = vertices[src.indices[2] as usize].pos;
            let normal = (v2 - v0).cross(v1 - v0).normalized();
            if !normal.x.is_nan() && !normal.y.is_nan() && !normal.z.is_nan() {
                // TODO: weight by angle at vertex?
                normals[src.indices[0] as usize] += normal;
                normals[src.indices[1] as usize] += normal;
                normals[src.indices[2] as usize] += normal;
            }
        }
        self.triangle_count = faces.len() as u32;

        let attribute_buffer_desc = BufferDesc::new(vertices.len() * mem::size_of::<AttributeData>());
        let mut writer = allocator
            .map_buffer(
                self.attribute_buffer,
                &attribute_buffer_desc,
                if with_ray_tracing {
                    BufferUsage::VERTEX_BUFFER | BufferUsage::RAY_TRACING_STORAGE_READ
                } else {
                    BufferUsage::VERTEX_BUFFER
                },
            )
            .unwrap();
        for src in normals.iter() {
            writer.write(src);
        }

        let scale = 0.9 / (max - min).component_max();
        let offset = (-0.5 * scale) * (max + min);
        for i in 0..Self::INSTANCE_COUNT {
            let corner = |i: usize, b| if ((i >> b) & 1usize) != 0usize { 0.5 } else { -0.5 };
            self.instances[i] = Similarity3::new(
                offset + Vec3::new(corner(i, 0), corner(i, 1), corner(i, 2)),
                Rotor3::identity(),
                scale,
            );
        }

        let instance_buffer_desc = BufferDesc::new(Self::INSTANCE_COUNT * mem::size_of::<InstanceData>());
        let mut writer = allocator
            .map_buffer(self.instance_buffer, &instance_buffer_desc, BufferUsage::VERTEX_BUFFER)
            .unwrap();
        for src in self.instances.iter() {
            let instance_data = src.into_homogeneous_matrix().into_transposed_transform();
            writer.write(&instance_data);
        }
    }
}
