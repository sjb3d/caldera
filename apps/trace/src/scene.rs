use caldera::*;

#[derive(Debug, Default)]
pub struct Transform(pub Similarity3);

#[derive(Debug)]
pub enum Geometry {
    TriangleMesh { positions: Vec<Vec3>, indices: Vec<UVec3> },
    Quad { transform: Similarity3, size: Vec2 },
    Sphere { centre: Vec3, radius: f32 },
}

#[derive(Debug)]
pub enum Surface {
    Diffuse { reflectance: Vec3 },
    Mirror { reflectance: f32 },
}

/*
    Ideally this would be some kind of shader that reads some
    interpolated data from the geometry (e.g. texture coordinates)
    and uniform data from the instance (e.g. textures, constants)
    and produces a closure for the BRDF (and emitter if present).

    For now we just enumerate some fixed options for the result of
    this shader.
*/
#[derive(Debug)]
pub struct Shader {
    pub surface: Surface,
    pub emission: Option<Vec3>,
}

#[derive(Debug)]
pub struct Instance {
    pub transform_ref: TransformRef,
    pub geometry_ref: GeometryRef,
    pub shader_ref: ShaderRef,
}

#[derive(Debug)]
pub struct Light {
    pub emission: Vec3,
}

#[derive(Debug)]
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
pub struct LightRef(pub u32);

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

#[derive(Debug, Default)]
pub struct Scene {
    pub transforms: Vec<Transform>,
    pub geometries: Vec<Geometry>,
    pub shaders: Vec<Shader>,
    pub instances: Vec<Instance>,
    pub lights: Vec<Light>,
    pub cameras: Vec<Camera>,
}

impl Scene {
    pub fn add_transform(&mut self, transform: Transform) -> TransformRef {
        let index = self.transforms.len();
        self.transforms.push(transform);
        TransformRef(index as u32)
    }

    pub fn add_geometry(&mut self, geometry: Geometry) -> GeometryRef {
        let index = self.geometries.len();
        self.geometries.push(geometry);
        GeometryRef(index as u32)
    }

    pub fn add_shader(&mut self, shader: Shader) -> ShaderRef {
        let index = self.shaders.len();
        self.shaders.push(shader);
        ShaderRef(index as u32)
    }

    pub fn add_instance(&mut self, instance: Instance) -> InstanceRef {
        let index = self.instances.len();
        self.instances.push(instance);
        InstanceRef(index as u32)
    }

    pub fn add_light(&mut self, light: Light) -> LightRef {
        let index = self.lights.len();
        self.lights.push(light);
        LightRef(index as u32)
    }

    pub fn add_camera(&mut self, camera: Camera) -> CameraRef {
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

    pub fn bake_unique_geometry(&mut self) {
        let mut instance_counts = vec![0u32; self.geometries.len()];
        for instance in self.instances.iter() {
            instance_counts[instance.geometry_ref.0 as usize] += 1;
        }
        let identity_ref = self.add_transform(Transform::default());
        for instance in self.instances.iter_mut() {
            if instance_counts[instance.geometry_ref.0 as usize] == 1 {
                let bake = self.transforms.get(instance.transform_ref.0 as usize).unwrap().0;
                match self.geometries.get_mut(instance.geometry_ref.0 as usize).unwrap() {
                    Geometry::TriangleMesh { positions, .. } => {
                        for pos in positions.iter_mut() {
                            *pos = bake * *pos;
                        }
                    }
                    Geometry::Quad { transform, .. } => {
                        *transform = bake * *transform;
                    }
                    Geometry::Sphere { centre, radius } => {
                        *centre = bake * *centre;
                        *radius = bake.scale.abs() * *radius;
                    }
                }
                instance.transform_ref = identity_ref;
            }
        }
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

pub struct ShaderBuilder(Shader);

impl ShaderBuilder {
    pub fn new_diffuse(reflectance: Vec3) -> Self {
        Self(Shader {
            surface: Surface::Diffuse {
                reflectance: reflectance.saturated(),
            },
            emission: None,
        })
    }

    pub fn new_mirror(reflectance: f32) -> Self {
        Self(Shader {
            surface: Surface::Mirror { reflectance },
            emission: None,
        })
    }

    pub fn with_emission(mut self, emission: Vec3) -> Self {
        let emission = emission.max_by_component(Vec3::zero());
        if emission.as_slice().iter().any(|&c| c > 0.0) {
            self.0.emission = Some(emission);
        }
        self
    }

    pub fn build(self) -> Shader {
        self.0
    }
}

struct SampledSpectrum {
    wavelength: f32,
    value: f32,
}
macro_rules! spectrum_samples {
    ($(($w:literal, $v:literal)),+) => { [ $( SampledSpectrum { wavelength: $w, value: $v, }, )+ ] }
}

// reference: https://www.graphics.cornell.edu/online/box/data.html
#[rustfmt::skip]
const CORNELL_BOX_WHITE_SAMPLES: &[SampledSpectrum] = &spectrum_samples!(
    (400.0, 0.343),(404.0, 0.445),(408.0, 0.551),(412.0, 0.624),(416.0, 0.665),
    (420.0, 0.687),(424.0, 0.708),(428.0, 0.723),(432.0, 0.715),(436.0, 0.710),
    (440.0, 0.745),(444.0, 0.758),(448.0, 0.739),(452.0, 0.767),(456.0, 0.777),
    (460.0, 0.765),(464.0, 0.751),(468.0, 0.745),(472.0, 0.748),(476.0, 0.729),
    (480.0, 0.745),(484.0, 0.757),(488.0, 0.753),(492.0, 0.750),(496.0, 0.746),
    (500.0, 0.747),(504.0, 0.735),(508.0, 0.732),(512.0, 0.739),(516.0, 0.734),
    (520.0, 0.725),(524.0, 0.721),(528.0, 0.733),(532.0, 0.725),(536.0, 0.732),
    (540.0, 0.743),(544.0, 0.744),(548.0, 0.748),(552.0, 0.728),(556.0, 0.716),
    (560.0, 0.733),(564.0, 0.726),(568.0, 0.713),(572.0, 0.740),(576.0, 0.754),
    (580.0, 0.764),(584.0, 0.752),(588.0, 0.736),(592.0, 0.734),(596.0, 0.741),
    (600.0, 0.740),(604.0, 0.732),(608.0, 0.745),(612.0, 0.755),(616.0, 0.751),
    (620.0, 0.744),(624.0, 0.731),(628.0, 0.733),(632.0, 0.744),(636.0, 0.731),
    (640.0, 0.712),(644.0, 0.708),(648.0, 0.729),(652.0, 0.730),(656.0, 0.727),
    (660.0, 0.707),(664.0, 0.703),(668.0, 0.729),(672.0, 0.750),(676.0, 0.760),
    (680.0, 0.751),(684.0, 0.739),(688.0, 0.724),(692.0, 0.730),(696.0, 0.740),
    (700.0, 0.737)
);
#[rustfmt::skip]
const CORNELL_BOX_GREEN_SAMPLES: &[SampledSpectrum] = &spectrum_samples!(
    (400.0, 0.092),(404.0, 0.096),(408.0, 0.098),(412.0, 0.097),(416.0, 0.098),
    (420.0, 0.095),(424.0, 0.095),(428.0, 0.097),(432.0, 0.095),(436.0, 0.094),
    (440.0, 0.097),(444.0, 0.098),(448.0, 0.096),(452.0, 0.101),(456.0, 0.103),
    (460.0, 0.104),(464.0, 0.107),(468.0, 0.109),(472.0, 0.112),(476.0, 0.115),
    (480.0, 0.125),(484.0, 0.140),(488.0, 0.160),(492.0, 0.187),(496.0, 0.229),
    (500.0, 0.285),(504.0, 0.343),(508.0, 0.390),(512.0, 0.435),(516.0, 0.464),
    (520.0, 0.472),(524.0, 0.476),(528.0, 0.481),(532.0, 0.462),(536.0, 0.447),
    (540.0, 0.441),(544.0, 0.426),(548.0, 0.406),(552.0, 0.373),(556.0, 0.347),
    (560.0, 0.337),(564.0, 0.314),(568.0, 0.285),(572.0, 0.277),(576.0, 0.266),
    (580.0, 0.250),(584.0, 0.230),(588.0, 0.207),(592.0, 0.186),(596.0, 0.171),
    (600.0, 0.160),(604.0, 0.148),(608.0, 0.141),(612.0, 0.136),(616.0, 0.130),
    (620.0, 0.126),(624.0, 0.123),(628.0, 0.121),(632.0, 0.122),(636.0, 0.119),
    (640.0, 0.114),(644.0, 0.115),(648.0, 0.117),(652.0, 0.117),(656.0, 0.118),
    (660.0, 0.120),(664.0, 0.122),(668.0, 0.128),(672.0, 0.132),(676.0, 0.139),
    (680.0, 0.144),(684.0, 0.146),(688.0, 0.150),(692.0, 0.152),(696.0, 0.157),
    (700.0, 0.159)
);
#[rustfmt::skip]
const CORNELL_BOX_RED_SAMPLES: &[SampledSpectrum] = &spectrum_samples!(
    (400.0, 0.040),(404.0, 0.046),(408.0, 0.048),(412.0, 0.053),(416.0, 0.049),
    (420.0, 0.050),(424.0, 0.053),(428.0, 0.055),(432.0, 0.057),(436.0, 0.056),
    (440.0, 0.059),(444.0, 0.057),(448.0, 0.061),(452.0, 0.061),(456.0, 0.060),
    (460.0, 0.062),(464.0, 0.062),(468.0, 0.062),(472.0, 0.061),(476.0, 0.062),
    (480.0, 0.060),(484.0, 0.059),(488.0, 0.057),(492.0, 0.058),(496.0, 0.058),
    (500.0, 0.058),(504.0, 0.056),(508.0, 0.055),(512.0, 0.056),(516.0, 0.059),
    (520.0, 0.057),(524.0, 0.055),(528.0, 0.059),(532.0, 0.059),(536.0, 0.058),
    (540.0, 0.059),(544.0, 0.061),(548.0, 0.061),(552.0, 0.063),(556.0, 0.063),
    (560.0, 0.067),(564.0, 0.068),(568.0, 0.072),(572.0, 0.080),(576.0, 0.090),
    (580.0, 0.099),(584.0, 0.124),(588.0, 0.154),(592.0, 0.192),(596.0, 0.255),
    (600.0, 0.287),(604.0, 0.349),(608.0, 0.402),(612.0, 0.443),(616.0, 0.487),
    (620.0, 0.513),(624.0, 0.558),(628.0, 0.584),(632.0, 0.620),(636.0, 0.606),
    (640.0, 0.609),(644.0, 0.651),(648.0, 0.612),(652.0, 0.610),(656.0, 0.650),
    (660.0, 0.638),(664.0, 0.627),(668.0, 0.620),(672.0, 0.630),(676.0, 0.628),
    (680.0, 0.642),(684.0, 0.639),(688.0, 0.657),(692.0, 0.639),(696.0, 0.635),
    (700.0, 0.642)
);
#[rustfmt::skip]
const CORNELL_BOX_LIGHT_SAMPLES: &[SampledSpectrum] = &spectrum_samples!(
    (400.0,  0.0),
    (500.0,  8.0),
    (600.0, 15.6),
    (700.0, 18.4),
    (750.0,  0.0)
);

fn xyz_from_samples(samples: &[SampledSpectrum]) -> Vec3 {
    let measure = |wavelength: f32| match samples
        .binary_search_by_key(&wavelength.to_bits(), |sample| sample.wavelength.to_bits())
    {
        Ok(index) => samples[index].value,
        Err(index) if 0 < index && index < samples.len() => {
            let s1 = unsafe { samples.get_unchecked(index) };
            let s0 = unsafe { samples.get_unchecked(index - 1) };
            assert!(s0.wavelength < wavelength && wavelength < s1.wavelength);
            let t = (wavelength - s0.wavelength) / (s1.wavelength - s0.wavelength);
            s0.value * (1.0 - t) + s1.value * t
        }
        _ => 0.0,
    };
    xyz_from_spectrum(measure) / xyz_from_spectrum(|_| 1.0).y
}

pub enum CornellBoxVariant {
    Original,
    Mirror,
    DomeLight,
    Instances,
    Sphere,
}

#[allow(clippy::excessive_precision)]
pub fn create_cornell_box_scene(variant: &CornellBoxVariant) -> Scene {
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
                Vec3::new(0.0, 0.5488, 0.0),
                Vec3::new(0.0, 0.5488, 0.5592),
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

    let rgb_from_xyz = xyz_from_rec709_matrix().inversed() * d65_from_e_matrix();
    let white_reflectance = rgb_from_xyz * xyz_from_samples(CORNELL_BOX_WHITE_SAMPLES);
    let red_reflectance = rgb_from_xyz * xyz_from_samples(CORNELL_BOX_RED_SAMPLES);
    let green_reflectance = rgb_from_xyz * xyz_from_samples(CORNELL_BOX_GREEN_SAMPLES);

    let white_shader = scene.add_shader(ShaderBuilder::new_diffuse(white_reflectance).build());
    let red_shader = scene.add_shader(ShaderBuilder::new_diffuse(red_reflectance).build());
    let green_shader = scene.add_shader(ShaderBuilder::new_diffuse(green_reflectance).build());
    let tall_block_shader = if matches!(variant, CornellBoxVariant::Mirror) {
        scene.add_shader(ShaderBuilder::new_mirror(1.0).build())
    } else {
        white_shader
    };

    let identity = scene.add_transform(Transform::default());

    scene.add_instance(Instance::new(identity, floor, white_shader));
    scene.add_instance(Instance::new(identity, ceiling, white_shader));
    scene.add_instance(Instance::new(identity, grey_wall, white_shader));
    scene.add_instance(Instance::new(identity, red_wall, red_shader));
    scene.add_instance(Instance::new(identity, green_wall, green_shader));
    if matches!(variant, CornellBoxVariant::Sphere) {
        let sphere = scene.add_geometry(Geometry::Sphere {
            centre: Vec3::new(0.15, 0.08, 0.15),
            radius: 0.08,
        });
        scene.add_instance(Instance::new(identity, sphere, white_shader));
    } else {
        scene.add_instance(Instance::new(identity, short_block, white_shader));
    }
    scene.add_instance(Instance::new(identity, tall_block, tall_block_shader));

    if matches!(variant, CornellBoxVariant::DomeLight) {
        scene.add_light(Light {
            emission: Vec3::new(0.4, 0.5, 0.6),
        });
    } else {
        let light_emission = rgb_from_xyz * xyz_from_samples(CORNELL_BOX_LIGHT_SAMPLES);
        let light_x0 = 0.213;
        let light_x1 = 0.343;
        let light_z0 = 0.227;
        let light_z1 = 0.332;
        let light_y = 0.5488 - 0.0001;
        let light_geometry = scene.add_geometry(Geometry::Quad {
            transform: Similarity3::new(
                Vec3::new(0.5 * (light_x1 + light_x0), light_y, 0.5 * (light_z1 + light_z0)),
                Rotor3::from_rotation_yz(0.5 * PI),
                1.0,
            ),
            size: Vec2::new(light_x1 - light_x0, light_z1 - light_z0),
        });
        let light_shader = scene.add_shader(
            ShaderBuilder::new_diffuse(Vec3::broadcast(0.78))
                .with_emission(light_emission)
                .build(),
        );
        scene.add_instance(Instance::new(identity, light_geometry, light_shader));
    }

    let camera_transform = scene.add_transform(Transform(Similarity3::new(
        Vec3::new(0.278, 0.273, -0.8),
        Rotor3::identity(),
        1.0,
    )));
    scene.add_camera(Camera {
        transform_ref: camera_transform,
        fov_y: 2.0 * (0.025f32 / 2.0).atan2(0.035),
    });

    if matches!(variant, CornellBoxVariant::Instances) {
        let extra_transforms: Vec<_> = (1..10)
            .map(|i| {
                let f = i as f32;
                scene.add_transform(Transform(Similarity3::new(
                    Vec3::new(-0.01 * f, 0.02 * f, 0.0),
                    Rotor3::from_rotation_xz(0.05 * f),
                    1.0,
                )))
            })
            .collect();

        for (i, extra) in extra_transforms.iter().cloned().enumerate() {
            scene.add_instance(Instance::new(
                extra,
                tall_block,
                if (i % 2) != 0 { green_shader } else { white_shader },
            ));
        }
        for (i, extra) in extra_transforms.iter().rev().cloned().enumerate() {
            scene.add_instance(Instance::new(
                extra,
                short_block,
                if (i % 2) != 0 { red_shader } else { white_shader },
            ));
        }
    }

    scene
}
