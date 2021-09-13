use bitvec::prelude::*;
use bytemuck::{Pod, Zeroable};
use caldera::prelude::*;

pub const MAX_VERTICES_PER_CLUSTER: usize = 64;
pub const MAX_TRIANGLES_PER_CLUSTER: usize = 124;

pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub triangles: Vec<UVec3>,
    pub face_normals: Vec<Vec3>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BoxBounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoxBounds {
    pub fn new() -> Self {
        Self {
            min: Vec3::broadcast(f32::MAX),
            max: Vec3::broadcast(f32::MIN),
        }
    }

    pub fn union_with_point(&mut self, p: Vec3) {
        self.min = self.min.min_by_component(p);
        self.max = self.max.max_by_component(p);
    }

    pub fn centre(&self) -> Vec3 {
        0.5 * (self.max + self.min)
    }
    pub fn half_extent(&self) -> Vec3 {
        0.5 * (self.max - self.min)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SphereBounds {
    pub centre: Vec3,
    pub radius: f32,
}

impl SphereBounds {
    pub fn from_box(aabb: BoxBounds) -> Self {
        Self {
            centre: aabb.centre(),
            radius: aabb.half_extent().mag(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ConeBounds {
    pub dir: Vec3,
    pub cos_angle: f32,
}

impl ConeBounds {
    pub fn from_box(aabb: BoxBounds) -> Self {
        // construct a cone that bounds the sphere
        let sphere = SphereBounds::from_box(aabb);
        let centre_dist = sphere.centre.mag();
        let sin_theta = sphere.radius / centre_dist;
        if sin_theta < 0.999 {
            Self {
                dir: sphere.centre.normalized(),
                cos_angle: (1.0 - sin_theta * sin_theta).max(0.0).sqrt(),
            }
        } else {
            Self {
                dir: Vec3::unit_x(),
                cos_angle: -1.0, // whole sphere
            }
        }
    }
}

#[derive(Debug)]
pub struct Cluster {
    pub position_bounds: BoxBounds,
    pub face_normal_bounds: BoxBounds,
    pub mesh_vertices: Vec<u32>,
    pub triangles: Vec<UVec3>,
    pub mesh_triangles: Vec<u32>,
}

impl Cluster {
    fn new() -> Self {
        Self {
            position_bounds: BoxBounds::new(),
            face_normal_bounds: BoxBounds::new(),
            mesh_vertices: Vec::new(),
            triangles: Vec::new(),
            mesh_triangles: Vec::new(),
        }
    }
}

struct TriangleListPerVertex {
    offset_per_vertex: Vec<u32>,
    triangle_indices: Vec<u32>,
}

impl TriangleListPerVertex {
    fn new(mesh: &Mesh) -> Self {
        let vertex_count = mesh.positions.len();
        let triangle_count = mesh.triangles.len();
        let corner_count = 3 * triangle_count;

        let mut offset_per_vertex = vec![0u32; vertex_count + 1];
        for triangle in mesh.triangles.iter() {
            for &vertex in triangle.as_slice() {
                offset_per_vertex[vertex as usize] += 1;
            }
        }

        let mut postfix_sum = 0u32;
        for offset in offset_per_vertex.iter_mut() {
            postfix_sum += *offset;
            *offset = postfix_sum;
        }

        let mut triangle_indices = vec![u32::MAX; corner_count];
        for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
            for &vertex in triangle.as_slice() {
                let offset_slot = &mut offset_per_vertex[vertex as usize];
                let offset = *offset_slot - 1;
                triangle_indices[offset as usize] = triangle_index as u32;
                *offset_slot = offset;
            }
        }

        Self {
            offset_per_vertex,
            triangle_indices,
        }
    }

    fn triangle_indices_for_vertex(&self, vertex: u32) -> &[u32] {
        let begin = self.offset_per_vertex[vertex as usize] as usize;
        let end = self.offset_per_vertex[vertex as usize + 1] as usize;
        &self.triangle_indices[begin..end]
    }
}

struct ClusterVertexRemap<'m> {
    mesh_positions: &'m [Vec3],
    position_bounds: BoxBounds,
    cluster_vertices: Vec<u8>,
}

impl<'m> ClusterVertexRemap<'m> {
    fn new(mesh: &'m Mesh) -> Self {
        Self {
            mesh_positions: &mesh.positions,
            position_bounds: BoxBounds::new(),
            cluster_vertices: vec![u8::MAX; mesh.positions.len()],
        }
    }

    fn get_or_insert(&mut self, mesh_vertex: u32, cluster: &mut Cluster) -> u32 {
        let mut cluster_vertex = self.cluster_vertices[mesh_vertex as usize];
        if cluster_vertex == u8::MAX {
            cluster_vertex = cluster.mesh_vertices.len() as u8;
            cluster.mesh_vertices.push(mesh_vertex);
            self.position_bounds
                .union_with_point(self.mesh_positions[mesh_vertex as usize]);
            self.cluster_vertices[mesh_vertex as usize] = cluster_vertex as u8;
        }
        cluster_vertex as u32
    }

    fn contains(&self, mesh_vertex: u32) -> bool {
        self.cluster_vertices[mesh_vertex as usize] != u8::MAX
    }

    fn finish(&mut self, cluster: &mut Cluster) {
        for &mesh_vertex in &cluster.mesh_vertices {
            self.cluster_vertices[mesh_vertex as usize] = u8::MAX;
        }
        cluster.position_bounds = self.position_bounds;
        self.position_bounds = BoxBounds::new();
    }
}

struct ClusterBuilder<'m> {
    mesh: &'m Mesh,
    triangle_list_per_vertex: TriangleListPerVertex,
    available_triangles: BitVec,
    vertex_remap: ClusterVertexRemap<'m>,
}

impl<'m> ClusterBuilder<'m> {
    fn new(mesh: &'m Mesh) -> Self {
        Self {
            mesh,
            triangle_list_per_vertex: TriangleListPerVertex::new(mesh),
            available_triangles: bitvec![1; mesh.triangles.len()],
            vertex_remap: ClusterVertexRemap::new(mesh),
        }
    }

    fn find_next_triangle(&self, cluster: &Cluster) -> Option<u32> {
        // early out if full
        if cluster.mesh_vertices.len() == MAX_VERTICES_PER_CLUSTER
            || cluster.triangles.len() == MAX_TRIANGLES_PER_CLUSTER
        {
            return None;
        }

        // select any triangle if the cluster is empty
        if cluster.triangles.is_empty() {
            return self
                .available_triangles
                .first_one()
                .map(|triangle_index| triangle_index as u32);
        }

        // HACK: just return the first one for now
        // TODO: build score using effect on cluster position and normal bounds
        cluster
            .mesh_vertices
            .iter()
            .flat_map(|&vertex| self.triangle_list_per_vertex.triangle_indices_for_vertex(vertex))
            .copied()
            .filter(|&triangle_index| self.available_triangles[triangle_index as usize])
            .filter(|&triangle_index| {
                let new_vertex_count = self.mesh.triangles[triangle_index as usize]
                    .as_slice()
                    .iter()
                    .copied()
                    .filter(|&vertex| !self.vertex_remap.contains(vertex))
                    .count();
                cluster.mesh_vertices.len() + new_vertex_count <= MAX_VERTICES_PER_CLUSTER
            })
            .next()
    }

    fn build(mut self) -> Vec<Cluster> {
        let mut clusters = Vec::new();
        loop {
            let mut cluster = Cluster::new();
            while let Some(triangle_index) = self.find_next_triangle(&cluster) {
                let triangle = self.mesh.triangles[triangle_index as usize]
                    .map_mut(|mesh_vertex| self.vertex_remap.get_or_insert(mesh_vertex, &mut cluster));
                cluster.triangles.push(triangle);
                cluster.mesh_triangles.push(triangle_index);
                cluster
                    .face_normal_bounds
                    .union_with_point(self.mesh.face_normals[triangle_index as usize]);
                self.available_triangles.set(triangle_index as usize, false);
            }
            if cluster.triangles.is_empty() {
                break;
            }
            self.vertex_remap.finish(&mut cluster);
            clusters.push(cluster);
        }
        clusters
    }
}

pub fn build_clusters(mut mesh: Mesh) -> (Mesh, Vec<Cluster>) {
    let mut clusters = ClusterBuilder::new(&mesh).build();

    // re-order vertices and triangles to appear in the order they are referenced by cluster triangles
    let mut new_vertex_from_old = vec![u32::MAX; mesh.positions.len()];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    mesh.triangles.clear();
    for cluster in clusters.iter_mut() {
        for mesh_vertex in cluster.mesh_vertices.iter_mut() {
            let old_vertex = *mesh_vertex;
            let mut new_vertex = new_vertex_from_old[old_vertex as usize];
            if new_vertex == u32::MAX {
                new_vertex = positions.len() as u32;
                new_vertex_from_old[old_vertex as usize] = new_vertex;
                positions.push(mesh.positions[old_vertex as usize]);
                normals.push(mesh.normals[old_vertex as usize]);
            }
            *mesh_vertex = new_vertex;
        }
        cluster.mesh_triangles.clear();
        for &triangle in &cluster.triangles {
            cluster.mesh_triangles.push(mesh.triangles.len() as u32);
            mesh.triangles
                .push(triangle.map_mut(|cluster_vertex| cluster.mesh_vertices[cluster_vertex as usize]));
        }
    }
    mesh.positions = positions;
    mesh.normals = normals;

    (mesh, clusters)
}
