use bevy::{
    asset::{Assets, Handle},
    log::debug,
    render::render_resource::Shader,
    utils::HashMap,
};
use std::hash::Hash;

pub struct PipelineRegistry {
    cache: HashMap<String, Handle<Shader>>,
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        Self {
            cache: Default::default(),
        }
    }
}

impl PipelineRegistry {
    pub fn configure(&mut self, source: &str, shaders: &mut Assets<Shader>) -> Handle<Shader> {
        if let Some(handle) = self.cache.get(source) {
            handle.clone()
        } else {
            let handle = shaders.add(Shader::from_wgsl(source.to_string()));
            debug!("Inserted new configured shader: {:?}\n{}", handle, source);
            self.cache.insert(source.to_string(), handle.clone());
            handle
        }
    }
}
