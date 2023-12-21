import file_io as fio

def load_quaternion_dataset(qpth):
    if fio.file_exist(qpth) == False:
        return
    with open(qpth) as f:
        data = f.readlines()
        quaternion_dict = {}
        if len(data) < 1:
            return
        for rd_str in data:
            rd_str = rd_str.replace('\n', '')
            rd = rd_str.split(' ')
            if len(rd) < 8:
                continue
            tag = rd[0]
            qx = float(rd[1])
            qy = float(rd[2])
            qz = float(rd[3])
            qw = float(rd[4])
            tx = float(rd[5])
            ty = float(rd[6])
            tz = float(rd[7])
            quaternion_dict[tag] = [qx, qy, qz, qw, tx, ty, tz]
        return quaternion_dict


class Sequence():
    image_dict = dict()

    def __init__(self, seq_dir, seq_tag) -> None:
        files = fio.traverse_dir(seq_dir, full_path=True, towards_sub=False)
        image_paths = fio.filter_ext(files, filter_out_target=False, ext_set=fio.img_ext_set)
        for image_pth in image_paths:
            (imgdir, imgname, imgext) = fio.get_filename_components(image_pth)
            image_tag = '/'.join([seq_tag, imgname + '.' + imgext])
            self.image_dict[image_tag] = image_pth


class Scene():
    def __init__(self, scene_dir) -> None:
        components = fio.traverse_dir(scene_dir)
        sequences_tags = fio.filter_folder(components, filter_out=False, filter_text='seq')
        sequences_paths = [fio.createPath(fio.sep, [scene_dir], x) for x in sequences_tags]

        self.image_dict = {}
        for i in range(0, len(sequences_tags)):
            seq_dir = sequences_paths[i]
            seq_tag = sequences_tags[i]
            seq = Sequence(seq_dir, seq_tag)
            self.image_dict.update(seq.image_dict)

        quaternion_paths = [fio.createPath(fio.sep, [scene_dir], x) for x in components]
        quaternion_sets = fio.filter_ext(quaternion_paths, filter_out_target=False, ext_set=['txt'])
        trainset_path = ''
        testset_path = ''
        for qpth in quaternion_sets:
            (filedir, file, ext) = fio.get_filename_components(qpth)
            if 'train' in file:
                trainset_path = qpth
            elif 'test' in file:
                testset_path = qpth
        self.train_quater = load_quaternion_dataset(trainset_path)
        self.test_quater = load_quaternion_dataset(testset_path)
