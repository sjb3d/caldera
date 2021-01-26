use caldera::*;

#[derive(Default)]
pub struct Transform(Isometry3); // TODO: allow scale?

#[derive(Default)]
pub struct TriangleMesh {
    positions: Vec<Vec3>,
    indices: Vec<UVec3>,
}

impl TriangleMesh {
    fn with_quad(mut self, v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        let base = UVec3::broadcast(self.positions.len() as u32);
        self.positions.push(v0);
        self.positions.push(v1);
        self.positions.push(v2);
        self.positions.push(v3);
        self.indices.push(base + UVec3::new(0, 1, 2));
        self.indices.push(base + UVec3::new(2, 3, 0));
        self
    }
}

pub struct Shader {
    debug_color: Vec3,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransformIndex(u32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeometryIndex(u32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderIndex(u32);

pub struct GeometryInstance {
    transform: TransformIndex,
    geometry: GeometryIndex,
    shader: ShaderIndex,
}

impl GeometryInstance {
    fn new(transform: TransformIndex, geometry: GeometryIndex, shader: ShaderIndex) -> Self {
        Self {
            transform,
            geometry,
            shader,
        }
    }
}

#[derive(Default)]
pub struct Scene {
    transforms: Vec<Transform>,
    geometries: Vec<TriangleMesh>,
    shaders: Vec<Shader>,
    geometry_instances: Vec<GeometryInstance>,
}

impl Scene {
    fn add_transform(&mut self, transform: Transform) -> TransformIndex {
        let index = self.transforms.len() - 1;
        self.transforms.push(transform);
        TransformIndex(index as u32)
    }

    fn add_triangle_mesh(&mut self, mesh: TriangleMesh) -> GeometryIndex {
        let index = self.geometries.len() - 1;
        self.geometries.push(mesh);
        GeometryIndex(index as u32)
    }

    fn add_shader(&mut self, debug_color: Vec3) -> ShaderIndex {
        let index = self.shaders.len() - 1;
        self.shaders.push(Shader { debug_color });
        ShaderIndex(index as u32)
    }

    fn add_geometry_instance(&mut self, instance: GeometryInstance) {
        self.geometry_instances.push(instance);
    }
}

pub fn create_cornell_box_scene() -> Scene {
    let mut scene = Scene::default();

    let floor = scene.add_triangle_mesh(TriangleMesh::default().with_quad(
        Vec3::new(0.5528, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.5592),
        Vec3::new(0.5496, 0.0, 0.5592),
    ));
    let ceiling = scene.add_triangle_mesh(TriangleMesh::default().with_quad(
        Vec3::new(0.556, 0.5488, 0.0),
        Vec3::new(0.556, 0.5488, 0.5592),
        Vec3::new(0.0, 0.5488, 0.5592),
        Vec3::new(0.0, 0.5488, 0.0),
    ));
    let grey_wall = scene.add_triangle_mesh(TriangleMesh::default().with_quad(
        Vec3::new(0.5496, 0.0, 0.5592),
        Vec3::new(0.0, 0.0, 0.5592),
        Vec3::new(0.0, 0.5488, 0.5592),
        Vec3::new(0.556, 0.5488, 0.5592),
    ));
    let red_wall = scene.add_triangle_mesh(TriangleMesh::default().with_quad(
        Vec3::new(0.5528, 0.0, 0.0),
        Vec3::new(0.5496, 0.0, 0.5592),
        Vec3::new(0.556, 0.5488, 0.5592),
        Vec3::new(0.556, 0.5488, 0.0),
    ));
    let green_wall = scene.add_triangle_mesh(TriangleMesh::default().with_quad(
        Vec3::new(0.0, 0.0, 0.5592),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.548, 0.0),
        Vec3::new(0.0, 0.548, 0.5592),
    ));
    let short_block = scene.add_triangle_mesh(
        TriangleMesh::default()
            .with_quad(
                Vec3::new(0.130, 0.165, 0.065),
                Vec3::new(0.082, 0.165, 0.225),
                Vec3::new(0.240, 0.165, 0.272),
                Vec3::new(0.290, 0.165, 0.114),
            )
            .with_quad(
                Vec3::new(0.290, 0.0, 0.114),
                Vec3::new(0.290, 0.165, 0.114),
                Vec3::new(0.240, 0.165, 0.272),
                Vec3::new(0.240, 0.0, 0.272),
            )
            .with_quad(
                Vec3::new(0.130, 0.0, 0.065),
                Vec3::new(0.130, 0.165, 0.065),
                Vec3::new(0.290, 0.165, 0.114),
                Vec3::new(0.290, 0.0, 0.114),
            )
            .with_quad(
                Vec3::new(0.082, 0.0, 0.225),
                Vec3::new(0.082, 0.165, 0.225),
                Vec3::new(0.130, 0.165, 0.065),
                Vec3::new(0.130, 0.0, 0.065),
            )
            .with_quad(
                Vec3::new(0.240, 0.0, 0.272),
                Vec3::new(0.240, 0.165, 0.272),
                Vec3::new(0.082, 0.165, 0.225),
                Vec3::new(0.082, 0.0, 0.225),
            ),
    );
    let tall_block = scene.add_triangle_mesh(
        TriangleMesh::default()
            .with_quad(
                Vec3::new(0.423, 0.330, 0.247),
                Vec3::new(0.265, 0.330, 0.296),
                Vec3::new(0.314, 0.330, 0.456),
                Vec3::new(0.472, 0.330, 0.406),
            )
            .with_quad(
                Vec3::new(0.423, 0.0, 0.247),
                Vec3::new(0.423, 0.330, 0.247),
                Vec3::new(0.472, 0.330, 0.406),
                Vec3::new(0.472, 0.0, 0.406),
            )
            .with_quad(
                Vec3::new(0.472, 0.0, 0.406),
                Vec3::new(0.472, 0.330, 0.406),
                Vec3::new(0.314, 0.330, 0.456),
                Vec3::new(0.314, 0.0, 0.456),
            )
            .with_quad(
                Vec3::new(0.314, 0.0, 0.456),
                Vec3::new(0.314, 0.330, 0.456),
                Vec3::new(0.265, 0.330, 0.296),
                Vec3::new(0.265, 0.0, 0.296),
            )
            .with_quad(
                Vec3::new(0.265, 0.0, 0.296),
                Vec3::new(0.265, 0.330, 0.296),
                Vec3::new(0.423, 0.330, 0.247),
                Vec3::new(0.423, 0.0, 0.247),
            ),
    );

    let grey_shader = scene.add_shader(Vec3::new(0.730, 0.735, 0.729));
    let red_shader = scene.add_shader(Vec3::new(0.611, 0.058, 0.062));
    let green_shader = scene.add_shader(Vec3::new(0.117, 0.449, 0.115));

    let identity = scene.add_transform(Transform::default());

    scene.add_geometry_instance(GeometryInstance::new(identity, floor, grey_shader));
    scene.add_geometry_instance(GeometryInstance::new(identity, ceiling, grey_shader));
    scene.add_geometry_instance(GeometryInstance::new(identity, grey_wall, grey_shader));
    scene.add_geometry_instance(GeometryInstance::new(identity, red_wall, red_shader));
    scene.add_geometry_instance(GeometryInstance::new(identity, green_wall, green_shader));
    scene.add_geometry_instance(GeometryInstance::new(identity, short_block, grey_shader));
    scene.add_geometry_instance(GeometryInstance::new(identity, tall_block, grey_shader));

    scene
}
