    // force field acceleration: note that the particles do not have a mass as of yet,
    // or we could say that the particles all have a mass of one, which means F = 1 * a.
    var ff_acceleration: vec3<f32> = vec3<f32>(0.0); 
    var not_conformed_to_sphere: f32 = 1.0;

    var unit_p2p_conformed: vec3<f32> = vec3<f32>(0.0);
    var conforming_source: vec3<f32> = vec3<f32>(0.0);
    var conforming_radius: f32 = 0.0;

    for (var kk: i32 = 0; kk < 16; kk=kk+1) {
        // As soon as a field component has a null mass, skip it and all subsequent ones.
        // Is this better than not having the if statement in the first place?
        // Likely answer:
        // The if statement is probably good in this case because all the particles will encounter 
        // the same number of field components.
        if (spawner.force_field[kk].mass == 0.0) {
            break;
        }

        let particle_to_point_source = (*particle).position - spawner.force_field[kk].position;
        let distance = length(particle_to_point_source);
        let unit_p2p = normalize(particle_to_point_source) ;

        let min_dist_check = step(spawner.force_field[kk].min_radius, distance);
        let max_dist_check = 1.0 - step(spawner.force_field[kk].max_radius, distance);

        // this turns into 0 when the field is an attractor and the particle is inside the min_radius and the source
        // is an attractor.
        if (spawner.force_field[kk].conform_to_sphere > 0.5) {
            not_conformed_to_sphere = not_conformed_to_sphere 
                * max(min_dist_check, -(sign(spawner.force_field[kk].mass) - 1.0) / 2.0);

            unit_p2p_conformed = 
                unit_p2p_conformed 
                + (1.0 - not_conformed_to_sphere) 
                * unit_p2p 
                * (1.0 - min_dist_check);

            conforming_source = 
                conforming_source 
                + (1.0 - not_conformed_to_sphere) 
                * spawner.force_field[kk].position
                * (1.0 - min_dist_check);

            conforming_radius = conforming_radius 
                + (1.0 - not_conformed_to_sphere) 
                * spawner.force_field[kk].min_radius / 1.2
                * (1.0 - min_dist_check);
        }

        let point_source_force =             
            - unit_p2p
            * min_dist_check * max_dist_check
            * spawner.force_field[kk].mass / 
                (0.0000001 + pow(distance, spawner.force_field[kk].force_exponent));
        
        // if the particle is within the min_radius of a source, then forget about
        // the other sources and only use the conformed field, thus the "* min_dist_check"
        ff_acceleration =  ff_acceleration * min_dist_check + point_source_force;
    }

    // conform to a sphere of radius min_radius/2 by projecting the velocity vector
    // onto a plane that is tangent to the sphere.
    let eps = vec3<f32>(0.000001);
    let projected_on_sphere = (*particle).velocity - proj(unit_p2p_conformed + eps, (*particle).velocity + eps);
    let conformed_field = (1.0 - not_conformed_to_sphere) * normalize(projected_on_sphere) * length((*particle).velocity);

    // Euler integration
    (*particle).velocity = ((*particle).velocity + (spawner.accel  + ff_acceleration) * sim_params.dt) * not_conformed_to_sphere + conformed_field;

    // let temp_vPos = (*particle).position;
    (*particle).position = (*particle).position + (*particle).velocity * sim_params.dt;
    
    // project on the sphere if within conforming distance
    let pos_to_source = conforming_source - (*particle).position;
    let difference = length(pos_to_source) - conforming_radius;
    (*particle).position = (*particle).position  + difference * normalize(pos_to_source) * (1.0 - not_conformed_to_sphere);

    // // commented because of the potential bug where dt could be zero, although the simulation
    // // works anyways, needs investigation
    // (*particle).velocity = ((*particle).position - temp_vPos) / sim_params.dt;