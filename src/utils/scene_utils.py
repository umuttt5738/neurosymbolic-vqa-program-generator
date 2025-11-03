import json
import numpy as np
from typing import List, Dict, Any


def _parse_scene_object(
    object_data: Dict[str, Any],
    scene_directions: Dict[str, List[float]],
    image_index: int,
    object_index: int,
) -> Dict[str, Any]:
    """
    Helper function to parse a single object's data from the scene file.
    Calculates 3D position if '3d_coords' is present.
    """
    scene_object = {}
    scene_object["id"] = f"{image_index}-{object_index}"

    # Handle object position
    if "3d_coords" in object_data:
        # Project 3D coordinates onto the scene's axes
        coords_3d = object_data["3d_coords"]
        right = scene_directions["right"]
        front = scene_directions["front"]
        
        # [x, y, z] -> [x_proj, y_proj, z_original]
        scene_object["position"] = [
            np.dot(coords_3d, right),
            np.dot(coords_3d, front),
            coords_3d[2],  # Z-coordinate remains the same
        ]
    else:
        # Fallback to 2D 'position' if '3d_coords' is missing
        scene_object["position"] = object_data["position"]

    # Copy other attributes
    scene_object["color"] = object_data["color"]
    scene_object["material"] = object_data["material"]
    scene_object["shape"] = object_data["shape"]
    scene_object["size"] = object_data["size"]

    return scene_object


def load_scenes(scenes_json_path: str) -> List[List[Dict[str, Any]]]:
    """
    Loads and parses the CLEVR scene JSON file.

    Args:
        scenes_json_path: The file path to the CLEVR_scenes.json file.

    Returns:
        A list of scenes. Each scene is a list of object dictionaries.
    """
    try:
        with open(scenes_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scene file not found at {scenes_json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {scenes_json_path}")
        return []

    all_scenes = []
    for scene_data in data["scenes"]:
        current_scene_objects = []
        scene_directions = scene_data.get("directions", {})
        image_index = scene_data["image_index"]

        for i, object_data in enumerate(scene_data["objects"]):
            scene_object = _parse_scene_object(
                object_data, scene_directions, image_index, i
            )
            current_scene_objects.append(scene_object)
        
        all_scenes.append(current_scene_objects)

    return all_scenes
