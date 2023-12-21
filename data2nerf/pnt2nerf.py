import numpy as np
import file_io as fio
from scene import Scene


source_dir = '/Users/siyanhu/Documents/posenet/HongKong'
scene_names = fio.traverse_dir(source_dir)

target_dir = '/Users/siyanhu/Documents/posenet/HongKong_nerf'
fio.ensure_dir(target_dir)


def quaternion_to_transformation_matrix(quater):
    q0 = quater[0]
    q1 = quater[1]
    q2 = quater[2]
    q3 = quater[3]
    tx = quater[4]
    ty = quater[5]
    tz = quater[6]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 4x4 rotation matrix
    trans_matrix = [[r00, r01, r02, tx],
                    [r10, r11, r12, ty],
                    [r20, r21, r22, tz],
                    [0, 0, 0, 1]]
    return trans_matrix


def create_frame_dict(image_paths_dict, dataset):
    keys = list(dataset.keys())
    frame_list = {}
    for k in keys:
        current_image_path = image_paths_dict[k]
        new_image_path = fio.createPath(fio.sep, [target_dir], k)
        if not fio.file_exist(new_image_path):
            (filedir, filename, fileext) = fio.get_filename_components(new_image_path)
            fio.ensure_dir(filedir)
            fio.copy_file(current_image_path, new_image_path)

        quaternion = dataset[k]
        transformation_matrix = quaternion_to_transformation_matrix(quaternion)
        frame = {'file_path': k, 'sharpness':0.5, 'transform_matrix': transformation_matrix}
        print(frame)
        break


for scene_tag in scene_names:
    scene_dir = fio.createPath(fio.sep, [source_dir], scene_tag)
    if fio.file_exist(scene_dir) == False:
        continue
    new_scene = Scene(scene_dir)

    image_paths_dict = new_scene.image_dict
    train_dataset = new_scene.train_quater
    test_dataset = new_scene.test_quater
    create_frame_dict(image_paths_dict, train_dataset)
    break