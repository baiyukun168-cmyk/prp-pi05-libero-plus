"""Build fixed complete Camera selections from LIBERO-Plus classification metadata."""


def select_camera(metadata, suite):
    if suite not in ("libero_spatial","libero_object","libero_goal","libero_10"):
        raise ValueError("Unsupported LIBERO-Plus suite")
    selected=[]
    for index,row in enumerate(metadata[suite]):
        if int(row["id"])-1 != index:
            raise ValueError("Classification IDs must match suite task order")
        if row["category"]=="Camera Viewpoints":
            selected.append({"task_id":index,"official_name":row["name"],"difficulty":int(row["difficulty_level"])})
    if not selected:
        raise ValueError("Camera selection is empty")
    return selected
