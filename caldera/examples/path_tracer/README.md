# path_tracer

This is an implementation of a spectral path tracer that makes use of Vulkan ray tracing extensions via [caldera](https://github.com/sjb3d/caldera).

## Features

* A uni-directional spectral path tracer
  * Currently samples 3 wavelengths per ray
  * Implemented as a single Vulkan ray tracing pipeline
  * Support for instanced geometry (via instanced bottom-level acceleration structures)
* Sampling using either pmj02 or sobol sequences (see links below)
* BSDF importance sampling
  * Diffuse and mirror "ideal" surfaces
  * Smooth or rough fresnel dieletrics and conductors
  * Diffuse with dielectric coating
* Importance sampling of lights
  * Quad/disc/sphere or triangle mesh shaped emitters
  * Dome or solid angle distant lights
* Multiple importance sampling between BSDFs and lights
* Simple fixed material model
  * Reflectance from per-instance constant and/or texture
  * All other parameters are either per-instance or global constants (for now)
* Interactive renderer with moveable camera and debug UI

The implementation makes use of the following Vulkan extensions:

- `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure` for tracing of rays. For now rays are traced using a single pipeline with all shading/sampling/traversal for a full path with multiple bounces.
  - The pipeline contains several different intersection and hit shaders to handle ray tracing against analytic shapes such as spheres and discs in addition to triangle meshes.
- `VK_KHR_buffer_device_address` to reference a GPU buffer using a `uint64_t` device address. This extension is required by `VK_KHR_ray_tracing_pipeline` for the shader binding table and vastly simplifies the setup of buffers, avoiding a lot of descriptor management code. The device address can be cast to a buffer reference of any type, so this can be useful to implement more generic GPU data structures.
- `VK_EXT_descriptor_indexing` for "bindless" texturing. This extension is also required by `VK_KHR_ray_tracing_pipeline` and allows us to bundle all textures into a single descriptor set with an array per texture type, and refer to them by index when they need to be sampled. Since different GPU threads can require different texture indices during ray tracing, we additionally make use of "non-uniform indexing" in the shader code.

## Links

Here are some of the references used when creating the renderer:

* [Hero Wavelength Spectral Sampling](https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14HWSS.pdf) to associate more than one wavelength with each ray (where possible) for reduced colour noise
* [Continuous Multiple Importance Sampling](http://iliyan.com/publications/ContinuousMIS) to sample all wavelengths of the ray proportional to the illuminant, to reduce colour noise even more
* [CIE 1931 Colour Matching Functions](http://cvrl.ioo.ucl.ac.uk/) to convert spectral samples to the XYZ colour space
* [CIE Standard Illuminants](https://www.rit.edu/cos/colorscience/rc_useful_data.php) has spectral data for lights
* [An RGB to Spectrum Conversion for Reflectances](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.40.9608&rep=rep1&type=pdf) to use Rec709/sRGB colours as reflectance in a spectral renderer
* [RefractiveIndex.info](https://refractiveindex.info/) for measured material properties for dielectrics and conductors
* [Chromatic Adaptation](http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html) description by Bruce Lindbloom to adjust the white point of XYZ samples
* [Filmic Tonemap Operators](http://filmicworlds.com/blog/filmic-tonemapping-operators/) for a Rec709/sRGB tonemap by Hejl and Burgess-Dawson
* [BakingLab](https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl) for a fit of the [ACES](https://github.com/ampas/aces-dev) tonemap
* [Progressive Multi-Jittered Sample Sequences](https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/) for random sequences, implemented in the [pmj](https://github.com/sjb3d/pmj) crate
* [Practical Hash-based Owen Scrambling](http://www.jcgt.org/published/0009/04/01/) to shuffle/scramble sequences that use the [Sobol direction numbers by Joe/Kuo](https://web.maths.unsw.edu.au/~fkuo/sobol/)
* [Survey of Efficient Representations for Independent Unit Vectors](http://jcgt.org/published/0003/02/01/) covers many options for efficient encoding of unit vectors, the renderer uses octohedral encoding in 32 bits
* [Sampling the GGX Distribution of Visible Normals](http://jcgt.org/published/0007/04/01/) to sample rough surfaces (all rough surfaces in the renderer have GGX-distributed microfacets)
* [Memo On Fresnel Equations](https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/) for the equations of fresnel dielectrics and conductors
* [Misunderstanding Layering](http://c0de517e.blogspot.com/2019/08/misunderstanding-multilayering-diffuse.html) approximation for energy conservation in layered diffuse and specular materials
* The [Physically Based Rendering From Theory To Implementation](https://www.pbrt.org/) (PBRT) book, I have the second edition but the 3rd edition is available for free [online](http://www.pbr-book.org/)!

The following crates have been super useful in making the app:

* [ultraviolet](https://crates.io/crates/ultraviolet): maths library with nice API, used for vectors and transforms
* [bytemuck](https://crates.io/crates/bytemuck): safely alias data as bytes, used for copying to GPU buffers
* [imgui](https://crates.io/crates/imgui)/[imgui-winit-support](https://crates.io/crates/imgui-winit-support): rust API for the fantastic [Dear ImGui](https://github.com/ocornut/imgui), for debug UI
* [serde](https://crates.io/crates/serde)/[serde_json](https://crates.io/crates/serde_json): amazing lib to generate serialisation for rust data structures, used to load Tungsten format JSON scenes
* [stb](https://crates.io/crates/stb): rust API for the [stb libraries](https://github.com/nothings/stb), used for image IO and BC compression
* [winit](https://crates.io/crates/winit): cross platform windowing and events

## How To Run

The code can be run as follows (will show a Cornell box by default):

```
make && cargo run --release --example path_tracer --
```

By default this will create a window with a progressive renderer and debug UI for many parameters.  Additionally you can drag with the mouse and use W/A/S/D on the keyboard to move the camera.  For command-line help run as:

```
make && cargo run --release --example path_tracer -- help
```

Several of the images below are loaded from [Tungsten](https://github.com/tunabrain/tungsten) format scenes.  These can be loaded into the renderer by running using the commandline:

```
make && cargo run --release --example path_tracer -- tungsten <scene_json_file_name>
```

## Test Images

As is tradition, here are some boxes under a couple of different lighting conditions (original [Cornell box](https://www.graphics.cornell.edu/online/box/data.html) data, and a variant with a mirror material and distant lights).

![cornell-box](images/cornell-box.jpg) ![cornell-box_dome-light](images/cornell-box_dome-light.jpg)

Here is a variation on the classic Veach multiple importance sampling scene, showing 64 samples per pixel with BSDF sampling only, 64 with light sampling only, then 32 samples of each weighted using multiple importance sampling.
These images demonstrate how multiple importance sampling effectively combines BSDF and light sampling to reduce variance over the whole image.

 BSDF Sampling Only | Light Sampling Only | Combine with MIS
:---: | :---: | :---:
![cornell-box_conductor_surfaces-only](images/cornell-box_conductor_surfaces-only.jpg) | ![cornell-box_conductor_lights-only](images/cornell-box_conductor_lights-only.jpg) | ![cornell-box_conductor](images/cornell-box_conductor.jpg)

Here is a test scene for some conductors using spectral reflectance data from [refractiveindex.info](https://refractiveindex.info/) for copper, iron and gold under a uniform illuminant (the colours are entirely from the reflectance data, there is no additional tinting).

![trace_material_conductors](images/material_conductors.jpg)

If we change the illuminant to F10 (which has a very spiky distribution), we can check the effect that wavelength importance sampling has on colour noise. The following images use gold lit with F10, all with 8 paths per pixel and 3 wavelengths per path. The first image samples wavelengths uniformly, the second samples only the hero wavelength for that path proportional to F10, the third image samples all wavelengths for that path proportional to F10 (reproducing part of the result of [Continuous Multiple Importance Sampling](http://iliyan.com/publications/ContinuousMIS)):

Uniform Sampling | Sample Hero Wavelength Only | Sample All Wavelengths
:---: | :---: | :---:
![trace_material_gold_f10_uniform](images/material_gold_f10_uniform.jpg) | ![trace_material_gold_f10_hero](images/material_gold_f10_hero.jpg) | ![trace_material_gold_f10_continuous](images/material_gold_f10_continuous.jpg)

## Gallery

The next set of images are rendered from these excellent [rendering resources](https://benedikt-bitterli.me/resources/) by Benedikt Bitterli and [blendswap.com](https://blendswap.com/) artists nacimus, Wig42, cekuhnen, Jay-Artist, thecali, NewSee2l035 and aXel.

![bathroom2](images/bathroom2.jpg)

![staircase](images/staircase.jpg) ![coffee](images/coffee.jpg)

![living-room-2](images/living-room-2.jpg)

![spaceship](images/spaceship.jpg)

![staircase2](images/staircase2.jpg)

![glass-of-water](images/glass-of-water.jpg)

There is a barely started exporter for Blender, but support for materials beyond a simple texture map is a bit out of scope for now.  This image uses the "Classroom" [Blender demo file](https://www.blender.org/download/demo-files/), with highly approximated materials and only sunlight:

![blender](images/blender.jpg)

## Potential Future Work

- [ ] Denoiser?
- [ ] Adaptive sampling
- [ ] HDR display output
- [x] Rough dielectrics
- [ ] Smooth conductors
- [ ] Generic clearcoat?
- [x] IOR parameters for conductors
- [ ] IOR parameters for dielectrics
- [ ] Interior media
- [x] Sobol sampler
- [x] Thin lens camera
- [ ] Volumetrics
- [ ] Image-based dome light
- [ ] More flexible materials (graphs?)
- [x] Disc primitive
- [x] Triangle mesh emitter?
- [ ] Microfacet multi-scattering?
- [ ] Path re-use?
- [x] Spectral rendering?
- [x] Spiky illuminants (F10)
