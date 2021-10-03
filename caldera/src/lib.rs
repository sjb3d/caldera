mod allocator;
mod app_base;
mod barrier;
mod color_space;
mod command_buffer;
mod context;
mod descriptor;
mod heap;
mod loader;
mod maths;
mod pipeline_cache;
mod query;
mod render_cache;
mod render_graph;
mod resource;
mod swapchain;
pub mod window_surface;

pub mod prelude {
    pub use caldera_macro::*;

    pub use crate::allocator::*;
    pub use crate::app_base::*;
    pub use crate::barrier::*;
    pub use crate::color_space::*;
    pub use crate::command_buffer::*;
    pub use crate::context::*;
    pub use crate::descriptor::*;
    pub use crate::loader::*;
    pub use crate::maths::*;
    pub use crate::pipeline_cache::*;
    pub use crate::query::*;
    pub use crate::render_cache::*;
    pub use crate::render_graph::*;
    pub use crate::resource::*;
    pub use crate::swapchain::*;

    pub use crate::command_name;
}
