import numpy as np

from src.transform.projection import CameraExtrinsics, CameraIntrinsics
from src.transform.projection.ray_caster import RayCaster


def test_round_trip_transformation():
    """Test that projecting a point to image and back to floor returns the original point."""
    # Setup simple camera: 2m high, looking down at 45 degrees
    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=640.0,
        cy=360.0,
        image_width=1280,
        image_height=720,
    )

    extrinsics = CameraExtrinsics.from_pose(
        camera_height_m=2.0,
        pitch_deg=45.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        camera_x_m=0.0,
        camera_y_m=0.0,
    )

    transformer = RayCaster(intrinsics, extrinsics)

    # Test point on floor: 2m in front of camera
    # Camera is at (0, 0, 2). Looking down 45 deg.
    # Forward direction is Y in World.
    # World: X-Right, Y-Forward, Z-Up.
    # Camera at (0, 0, 2).
    # Looking down 45 deg means optical axis points towards Y+ and Z-.

    # Let's pick a point on the floor (Z=0)
    # If pitch is 45 deg, the center of the image should look at:
    # tan(45) = 2 / Y => Y = 2m.
    # So (0, 2.0, 0) should project to center of image (640, 360).

    world_point = (0.0, 2.0, 0.0)
    pixel_point = transformer.floor_to_image(world_point)

    assert pixel_point is not None
    u, v = pixel_point
    assert abs(u - 640.0) < 1.0
    assert abs(v - 360.0) < 1.0

    # Now back to floor
    reconstructed_point = transformer.image_to_floor(pixel_point)
    assert reconstructed_point is not None
    rx, ry = reconstructed_point

    assert abs(rx - world_point[0]) < 0.01
    assert abs(ry - world_point[1]) < 0.01


def test_horizon_handling():
    """Test that points above horizon return None."""
    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=640.0,
        cy=360.0,
        image_width=1280,
        image_height=720,
    )

    # Camera looking straight forward (pitch=0)
    extrinsics = CameraExtrinsics.from_pose(
        camera_height_m=2.0,
        pitch_deg=0.0,  # Looking at horizon
        yaw_deg=0.0,
        roll_deg=0.0,
        camera_x_m=0.0,
        camera_y_m=0.0,
    )

    transformer = RayCaster(intrinsics, extrinsics)

    # Center of image should be at infinity (horizon)
    # Ideally it returns None or very large number
    # Our implementation checks for parallel ray or intersection behind camera

    # Point slightly above center (sky)
    pixel_point_sky = (640.0, 350.0)  # y < cy
    floor_point = transformer.image_to_floor(pixel_point_sky)
    assert floor_point is None


def test_distortion_correction():
    """Test that distortion is handled."""
    # Add some barrel distortion
    dist_coeffs = np.array([-0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=640.0,
        cy=360.0,
        image_width=1280,
        image_height=720,
        dist_coeffs=dist_coeffs,
    )

    extrinsics = CameraExtrinsics.from_pose(
        camera_height_m=2.0,
        pitch_deg=45.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        camera_x_m=0.0,
        camera_y_m=0.0,
    )

    transformer = RayCaster(intrinsics, extrinsics)

    # A point that would be distorted
    # We can't easily predict the exact pixel without calculation,
    # but we can verify round trip consistency.

    world_point = (1.0, 3.0, 0.0)
    pixel_point = transformer.floor_to_image(world_point)
    assert pixel_point is not None

    reconstructed_point = transformer.image_to_floor(pixel_point)
    assert reconstructed_point is not None

    rx, ry = reconstructed_point
    assert abs(rx - world_point[0]) < 0.05
    assert abs(ry - world_point[1]) < 0.05
