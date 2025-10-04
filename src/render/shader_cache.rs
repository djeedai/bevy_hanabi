use std::hash::BuildHasher;

use bevy::{
    asset::{Assets, Handle},
    ecs::{change_detection::ResMut, resource::Resource},
    log::{debug, trace},
    platform::collections::HashMap,
    shader::Shader,
};

/// Cache of baked shaders variants.
///
/// Baked shader variants are shaders where the placeholders `{{PLACEHOLDER}}`
/// present in the WGSL template code have been replaced by actual WGSL code,
/// making them a valid shader from the point of view of the Bevy renderer.
///
/// Shaders present in the cache are allocated [`Shader`] resources. Note that a
/// [`Shader`] resource _may_ further be preprocessed to replace `#define`
/// directives; to this extent, some entries may not be compilable WGSL as is.
#[derive(Default, Resource)]
pub struct ShaderCache {
    /// Map of allocated shader resources from their baked shader code.
    cache: HashMap<String, Handle<Shader>>,
}

impl ShaderCache {
    /// Get an existing baked shader variant, or insert it into the cache and
    /// allocate a new [`Shader`] resource for it.
    ///
    /// Returns the [`Shader`] resource associated with `source`.
    pub fn get_or_insert(
        &mut self,
        filename: &str,
        suffix: &str,
        source: &str,
        shaders: &mut ResMut<Assets<Shader>>,
    ) -> Handle<Shader> {
        if let Some(handle) = self.cache.get(source) {
            handle.clone()
        } else {
            let hash = bevy::platform::hash::FixedHasher.hash_one(source);
            let shader = Shader::from_wgsl(
                source.to_string(),
                format!("hanabi/{}_{}_{}.wgsl", filename, suffix, hash),
            );
            trace!(
                "Shader path={} import_path={:?} imports={:?}",
                shader.path,
                shader.import_path,
                shader.imports
            );
            let shader_path = shader.path.clone();
            let handle = shaders.add(shader);
            debug!(
                "Inserted new configured shader {}: {:?}\n{}",
                shader_path, handle, source
            );
            self.cache.insert(source.to_string(), handle.clone());
            handle
        }
    }
}
