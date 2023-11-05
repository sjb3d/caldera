use crate::{command_buffer::*, context::*};
use arrayvec::ArrayVec;
use spark::{vk, Device};
use std::{ffi::CStr, mem};

#[macro_export]
macro_rules! command_name {
    ($e:tt) => {
        unsafe { ::std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($e, "\0").as_bytes()) }
    };
}

#[derive(Debug)]
struct QuerySet {
    query_pool: vk::QueryPool,
    names: ArrayVec<Option<&'static CStr>, { Self::MAX_PER_FRAME as usize }>,
}

impl QuerySet {
    const MAX_PER_FRAME: u32 = 64;

    fn new(device: &Device) -> Self {
        let query_pool = {
            let create_info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: Self::MAX_PER_FRAME,
                ..Default::default()
            };

            unsafe { device.create_query_pool(&create_info, None) }.unwrap()
        };

        Self {
            query_pool,
            names: ArrayVec::new(),
        }
    }
}

pub struct QueryPool {
    context: SharedContext,
    sets: [QuerySet; Self::COUNT],
    last_us: ArrayVec<(&'static CStr, f32), { QuerySet::MAX_PER_FRAME as usize }>,
    timestamp_valid_mask: u64,
    timestamp_period_us: f32,
    index: usize,
    is_enabled: bool,
}

impl QueryPool {
    const COUNT: usize = CommandBufferPool::COUNT;

    pub fn new(context: &SharedContext) -> Self {
        let mut sets = ArrayVec::new();
        for _ in 0..Self::COUNT {
            sets.push(QuerySet::new(&context.device));
        }
        Self {
            context: SharedContext::clone(context),
            sets: sets.into_inner().unwrap(),
            last_us: ArrayVec::new(),
            timestamp_valid_mask: 1u64
                .checked_shl(context.queue_family_properties.timestamp_valid_bits)
                .unwrap_or(0)
                .wrapping_sub(1),
            timestamp_period_us: context.physical_device_properties.limits.timestamp_period / 1000.0,
            index: 0,
            is_enabled: true,
        }
    }

    pub fn begin_frame(&mut self, cmd: vk::CommandBuffer) {
        self.index = (self.index + 1) % Self::COUNT;
        self.last_us.clear();

        let set = &mut self.sets[self.index];
        if !set.names.is_empty() {
            let mut query_results = [0u64; QuerySet::MAX_PER_FRAME as usize];
            unsafe {
                self.context.device.get_query_pool_results(
                    set.query_pool,
                    0,
                    set.names.len() as u32,
                    &mut query_results,
                    mem::size_of::<u64>() as vk::DeviceSize,
                    vk::QueryResultFlags::N64 | vk::QueryResultFlags::WAIT,
                )
            }
            .unwrap();
            for (i, name) in set
                .names
                .iter()
                .take(QuerySet::MAX_PER_FRAME as usize - 1)
                .enumerate()
                .filter_map(|(i, name)| name.map(|name| (i, name)))
            {
                let timestamp_delta = (query_results[i + 1] - query_results[i]) & self.timestamp_valid_mask;
                self.last_us
                    .push((name, (timestamp_delta as f32) * self.timestamp_period_us));
            }
        }

        set.names.clear();

        unsafe {
            self.context
                .device
                .cmd_reset_query_pool(cmd, set.query_pool, 0, QuerySet::MAX_PER_FRAME)
        };
    }

    fn emit_timestamp_impl(&mut self, cmd: vk::CommandBuffer, name: Option<&'static CStr>) {
        let set = &mut self.sets[self.index];
        if self.is_enabled && !set.names.is_full() {
            unsafe {
                self.context.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    set.query_pool,
                    set.names.len() as u32,
                )
            };
            set.names.push(name);
        }
    }

    pub fn emit_timestamp(&mut self, cmd: vk::CommandBuffer, name: &'static CStr) {
        self.emit_timestamp_impl(cmd, Some(name));
    }

    pub fn end_frame(&mut self, cmd: vk::CommandBuffer) {
        self.emit_timestamp_impl(cmd, None);
    }

    pub fn ui_timestamp_table(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.is_enabled, "Enabled");
        egui::Grid::new("timestamp_grid").show(ui, |ui| {
            ui.label("Pass");
            ui.label("Time (us)");
            ui.end_row();
            for (name, time_us) in &self.last_us {
                ui.label(name.to_str().unwrap());
                ui.label(format!("{:>7.1}", time_us));
                ui.end_row();
            }
        });
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        for set in self.sets.iter() {
            unsafe {
                self.context.device.destroy_query_pool(Some(set.query_pool), None);
            }
        }
    }
}
