import os 
import shutil


for scene in os.listdir("/data/datasets/DAVIS-2016/DAVIS/JPEGImages/480p/"):
    print(scene)
    if os.path.exists("/data/users/mpilligua/hashing-nvd/data/" + scene):
        print("Scene already exists")
        continue
   
    # Create a directory for the scene
    os.makedirs("/data/users/mpilligua/hashing-nvd/data/" + scene, exist_ok=True)

    # Copy the images to the scene directory
    davis_dir = "/data/datasets/DAVIS-2016/DAVIS/"
    images_dir = davis_dir + "JPEGImages/480p/" + scene + "/"
    flow_dir = davis_dir + "flow/" + scene + "/"
    masks = davis_dir + "Annotations/480p/" + scene + "/"

    shutil.copytree(images_dir, "/data/users/mpilligua/hashing-nvd/data/" + scene + "/video_frames")
    shutil.copytree(flow_dir, "/data/users/mpilligua/hashing-nvd/data/" + scene + "/flow")
    shutil.copytree(masks, "/data/users/mpilligua/hashing-nvd/data/" + scene + "/gt_masks/folder1")

