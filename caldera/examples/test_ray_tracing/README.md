# test_ray_tracing

Run using:

```
make && cargo run --release --example test_ray_tracing -- <ply_file_name>
```

Example when run with the Stanford Bunny from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/) as `<ply_file_name>`.

![test_ray_tracing image](../../../images/test_ray_tracing.jpg)

The goal was to experiment with:

- Use the `VK_KHR_acceleration_structure` extension
  - Build a bottom-level acceleration structure for the PLY mesh
  - Build a top-level acceleration structure for the 8 instances
- Use the `VK_KHR_ray_tracing_pipeline` extension
  - Trace camera rays to the first hit in a compute shader
  - Just interpolate vertex normals at the hit point
- Compare with ray tracing and rasterization results and performance
- As an unrelated test, try out using a _transient attachment_ for MSAA depth
