from DartRobots.DartRobotsPy import Terrain, TerrainConfig, TerrainType, TerrainGenerator, get_height, World


def terrain_generator(terrain_type):
    generator = TerrainGenerator()

    # Configure the terrain

    config = TerrainConfig()

    # plane
    if terrain_type == "Plane":
        config.terrain_type = TerrainType.Plane
        config.x_size = 4.0
        config.y_size = 4.0
        config.resolution = 0.1

    # Hills
    elif terrain_type == "Hills":
        config.terrain_type = TerrainType.Hills
        config.x_size = 25.0
        config.y_size = 25.0
        config.resolution = 0.1
        config.seed = 566
        config.roughness = 0.001
        config.amplitude = 0.25
        config.frequency = 0.2
        config.num_octaves = 1

    # Steps
    elif terrain_type == "Steps":
        config.terrain_type = TerrainType.Steps
        config.x_size = 2.0
        config.y_size = 2.0
        config.resolution = 0.01
        config.step_width = 0.2
        config.step_height = 0.1

    # Generate the terrain
    print(terrain_type)
    terrain = generator.generate(config)

    return terrain
