use bevy::{
    render::{
        render_resource::{ComputePipeline, ComputePipelineDescriptor},
        renderer::RenderDevice,
    },
    utils::HashMap,
};
use std::hash::Hash;

/// A compute pipeline specialized by a key.
///
/// The compute pipeline is based on a source template, which is partially modified based
/// on the key to produce a new specialization (variant).
pub trait SpecializedComputePipeline {
    type Key: Clone + Hash + PartialEq + Eq;

    /// Specialize the base template pipeline with the given key.
    fn specialize(&self, key: Self::Key, render_device: &RenderDevice) -> ComputePipeline;
}

/// Cache for specialized (preprocessed) compute pipelines.
pub struct ComputeCache<S: SpecializedComputePipeline> {
    cache: HashMap<S::Key, ComputePipeline>,
}

impl<S: SpecializedComputePipeline> Default for ComputeCache<S> {
    fn default() -> Self {
        Self {
            cache: Default::default(),
        }
    }
}

impl<S: SpecializedComputePipeline> ComputeCache<S> {
    pub fn specialize(
        &mut self,
        pipeline: &S,
        key: S::Key,
        render_device: &RenderDevice,
    ) -> &ComputePipeline {
        self.cache
            .entry(key.clone())
            .or_insert_with(|| pipeline.specialize(key, render_device))
    }
}
