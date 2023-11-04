use bevy::{
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use noise::{NoiseFn, Perlin};

/// Create an animated sprite sheet texture.
///
/// Create a texture composed of individual sprites of size `size` pixels,
/// arranged into a grid of `grid` size into the texture image. The final image
/// has a pixel size of `size * grid`.
///
/// The texture is based on a 3D Perlin noise scaled with `scale.xy`. Each
/// sprite is a layer at a different height, scaled by `scale.z`, giving the
/// impression of animation.
///
/// The final image is convoluted by a falloff disk to ensure the square border
/// of the texture are hidden. Instead the texture appears roughly circular.
///
/// This produces an R8Unorm texture where the R component is equal to the
/// opacity, to be used with the [`ImageSampleMapping::ModulateOpacityFromR`]
/// mode of the [`ParticleTextureModifier`].
///
/// This code is a utility for examples. It's nowhere near efficient or clean as
/// could be for production.
pub fn make_anim_img(size: UVec2, grid: UVec2, scale: Vec3) -> Image {
    let w = Perlin::new(42);
    let tile_cols = size.x as usize;
    let tile_rows = size.y as usize;
    let grid_cols = grid.x as usize;
    let grid_rows = grid.y as usize;
    let tex_cols = tile_cols * grid_cols;
    let tex_rows = tile_rows * grid_rows;
    let tex_len = tex_cols * tex_rows;
    let mut data = vec![0; tex_len];
    let mut k = 0.;
    let dk = scale.z as f64;
    let tile_half_size = Vec2::new(size.x as f32 * scale.x, size.y as f32 * scale.y) / 2.;
    let tile_radius = tile_half_size.x.abs().max(tile_half_size.y.abs()) * 0.9;
    for v in 0..grid.y as usize {
        let index0 = v * tex_cols * tile_rows;
        for u in 0..grid.x as usize {
            let index1 = index0 + u * tile_cols;
            for j in 0..size.y as usize {
                let index2 = index1 + j * tex_cols;
                for i in 0..size.x as usize {
                    let index3 = index2 + i;
                    let pt = Vec2::new(i as f32 * scale.x, j as f32 * scale.y);
                    let dist = pt.distance(tile_half_size);
                    let falloff = (dist / tile_radius).clamp(0., 1.);
                    let value = w.get([pt.x as f64, pt.y as f64, k]) * 256.;
                    let falloff_value = value * (1.0 - falloff as f64);
                    let falloff_height = (falloff_value as u32).clamp(0, 255) as u8;
                    data[index3] = falloff_height;
                }
            }
            k += dk;
        }
    }
    Image::new(
        Extent3d {
            width: tex_cols as u32,
            height: tex_rows as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
    )
}
