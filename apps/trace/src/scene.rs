use caldera::*;

#[derive(Default)]
pub struct Transform(pub Isometry3); // TODO: allow scale?

pub enum Geometry {
    TriangleMesh { positions: Vec<Vec3>, indices: Vec<UVec3> },
    Quad { transform: Isometry3, size: Vec2 },
}

pub struct Shader {
    pub reflectance: Vec3,
    pub emission: Vec3,
}

impl Shader {
    pub fn new_lambertian(reflectance: Vec3) -> Self {
        Self {
            reflectance,
            emission: Vec3::zero(),
        }
    }

    pub fn with_emission(mut self, emission: Vec3) -> Self {
        self.emission = emission;
        self
    }

    pub fn is_emissive(&self) -> bool {
        self.emission.as_slice().iter().any(|&c| c > 0.0)
    }
}

pub struct Instance {
    pub transform_ref: TransformRef,
    pub geometry_ref: GeometryRef,
    pub shader_ref: ShaderRef,
}

pub struct Camera {
    pub transform_ref: TransformRef,
    pub fov_y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransformRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeometryRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstanceRef(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CameraRef(pub u32);

impl Instance {
    fn new(transform_ref: TransformRef, geometry_ref: GeometryRef, shader_ref: ShaderRef) -> Self {
        Self {
            transform_ref,
            geometry_ref,
            shader_ref,
        }
    }
}

#[derive(Default)]
pub struct Scene {
    pub transforms: Vec<Transform>,
    pub geometries: Vec<Geometry>,
    pub shaders: Vec<Shader>,
    pub instances: Vec<Instance>,
    pub cameras: Vec<Camera>,
}

impl Scene {
    fn add_transform(&mut self, transform: Transform) -> TransformRef {
        let index = self.transforms.len();
        self.transforms.push(transform);
        TransformRef(index as u32)
    }

    fn add_geometry(&mut self, geometry: Geometry) -> GeometryRef {
        let index = self.geometries.len();
        self.geometries.push(geometry);
        GeometryRef(index as u32)
    }

    fn add_shader(&mut self, shader: Shader) -> ShaderRef {
        let index = self.shaders.len();
        self.shaders.push(shader);
        ShaderRef(index as u32)
    }

    fn add_instance(&mut self, instance: Instance) -> InstanceRef {
        let index = self.instances.len();
        self.instances.push(instance);
        InstanceRef(index as u32)
    }

    fn add_camera(&mut self, camera: Camera) -> CameraRef {
        let index = self.cameras.len();
        self.cameras.push(camera);
        CameraRef(index as u32)
    }

    pub fn geometry_ref_iter(&self) -> impl Iterator<Item = GeometryRef> {
        (0..self.geometries.len()).map(|i| GeometryRef(i as u32))
    }

    pub fn instance_ref_iter(&self) -> impl Iterator<Item = InstanceRef> {
        (0..self.instances.len()).map(|i| InstanceRef(i as u32))
    }

    pub fn camera_ref_iter(&self) -> impl Iterator<Item = CameraRef> {
        (0..self.cameras.len()).map(|i| CameraRef(i as u32))
    }

    pub fn transform(&self, r: TransformRef) -> &Transform {
        self.transforms.get(r.0 as usize).unwrap()
    }

    pub fn geometry(&self, r: GeometryRef) -> &Geometry {
        self.geometries.get(r.0 as usize).unwrap()
    }

    pub fn shader(&self, r: ShaderRef) -> &Shader {
        self.shaders.get(r.0 as usize).unwrap()
    }

    pub fn instance(&self, r: InstanceRef) -> &Instance {
        self.instances.get(r.0 as usize).unwrap()
    }

    pub fn camera(&self, r: CameraRef) -> &Camera {
        self.cameras.get(r.0 as usize).unwrap()
    }
}

pub struct TriangleMeshBuilder {
    pub positions: Vec<Vec3>,
    pub indices: Vec<UVec3>,
}

impl TriangleMeshBuilder {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn with_quad(mut self, v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        let base = UVec3::broadcast(self.positions.len() as u32);
        self.positions.push(v0);
        self.positions.push(v1);
        self.positions.push(v2);
        self.positions.push(v3);
        self.indices.push(base + UVec3::new(0, 1, 2));
        self.indices.push(base + UVec3::new(2, 3, 0));
        self
    }

    pub fn build(self) -> Geometry {
        Geometry::TriangleMesh {
            positions: self.positions,
            indices: self.indices,
        }
    }
}

pub fn create_cornell_box_scene() -> Scene {
    let mut scene = Scene::default();

    let floor = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.5528, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.5592),
                Vec3::new(0.5496, 0.0, 0.5592),
            )
            .build(),
    );
    let ceiling = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.556, 0.5488, 0.0),
                Vec3::new(0.556, 0.5488, 0.5592),
                Vec3::new(0.0, 0.5488, 0.5592),
                Vec3::new(0.0, 0.5488, 0.0),
            )
            .build(),
    );
    let grey_wall = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.5496, 0.0, 0.5592),
                Vec3::new(0.0, 0.0, 0.5592),
                Vec3::new(0.0, 0.5488, 0.5592),
                Vec3::new(0.556, 0.5488, 0.5592),
            )
            .build(),
    );
    let red_wall = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.5528, 0.0, 0.0),
                Vec3::new(0.5496, 0.0, 0.5592),
                Vec3::new(0.556, 0.5488, 0.5592),
                Vec3::new(0.556, 0.5488, 0.0),
            )
            .build(),
    );
    let green_wall = scene.add_geometry(
        TriangleMeshBuilder::new()
            .with_quad(
                Vec3::new(0.0, 0.0, 0.5592),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.548, 0.0),
                Vec3::new(0.0, 0.548, 0.5592),
            )
            .build(),
    );
    let short_block = scene.add_geometry(
        TriangleMeshBuilder::new()
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
            )
            .build(),
    );
    let tall_block = scene.add_geometry(
        TriangleMeshBuilder::new()
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
            )
            .build(),
    );

    let grey_shader = scene.add_shader(Shader::new_lambertian(Vec3::new(0.730, 0.735, 0.729)));
    let red_shader = scene.add_shader(Shader::new_lambertian(Vec3::new(0.611, 0.058, 0.062)));
    let green_shader = scene.add_shader(Shader::new_lambertian(Vec3::new(0.117, 0.449, 0.115)));

    let identity = scene.add_transform(Transform::default());

    scene.add_instance(Instance::new(identity, floor, grey_shader));
    scene.add_instance(Instance::new(identity, ceiling, grey_shader));
    scene.add_instance(Instance::new(identity, grey_wall, grey_shader));
    scene.add_instance(Instance::new(identity, red_wall, red_shader));
    scene.add_instance(Instance::new(identity, green_wall, green_shader));
    scene.add_instance(Instance::new(identity, short_block, grey_shader));
    scene.add_instance(Instance::new(identity, tall_block, grey_shader));

    let light_x0 = 0.213;
    let light_x1 = 0.343;
    let light_z0 = 0.227;
    let light_z1 = 0.332;
    let light_y = 0.5488 - 0.0001;
    let light_geometry = scene.add_geometry(Geometry::Quad {
        transform: Isometry3::new(
            Vec3::new(0.5 * (light_x1 + light_x0), light_y, 0.5 * (light_z1 + light_z0)),
            Rotor3::from_rotation_yz(0.5 * PI),
        ),
        size: Vec2::new(light_x1 - light_x0, light_z1 - light_z0),
    });
    let light_shader =
        scene.add_shader(Shader::new_lambertian(Vec3::broadcast(0.78)).with_emission(Vec3::new(17.0, 11.8, 4.0)));
    scene.add_instance(Instance::new(identity, light_geometry, light_shader));

    let camera_transform = scene.add_transform(Transform(Isometry3::new(
        Vec3::new(0.278, 0.273, -0.8),
        Rotor3::identity(),
    )));
    scene.add_camera(Camera {
        transform_ref: camera_transform,
        fov_y: 2.0 * (0.025f32 / 2.0).atan2(0.035),
    });

    let add_extra_blocks = false;
    if add_extra_blocks {
        let extra_transforms: Vec<_> = (1..10)
            .map(|i| {
                let f = i as f32;
                scene.add_transform(Transform(Isometry3::new(
                    Vec3::new(-0.01 * f, 0.02 * f, 0.0),
                    Rotor3::from_rotation_xz(0.05 * f),
                )))
            })
            .collect();

        for (i, extra) in extra_transforms.iter().cloned().enumerate() {
            scene.add_instance(Instance::new(
                extra,
                tall_block,
                if (i % 2) != 0 { green_shader } else { grey_shader },
            ));
        }
        for (i, extra) in extra_transforms.iter().rev().cloned().enumerate() {
            scene.add_instance(Instance::new(
                extra,
                short_block,
                if (i % 2) != 0 { red_shader } else { grey_shader },
            ));
        }
    }

    scene
}
