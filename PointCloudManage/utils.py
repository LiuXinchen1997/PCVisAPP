import os
from time import time
import zipfile
import numpy as np
import open3d as o3d
import tensorflow as tf

from plyfile import PlyData, PlyElement

from PointCloudManage.exception import ShapeException, FileExistsException
from PointCloudManage.upsample_op.config import CFG
from PointCloudManage.upsample_op.model import Model


# others
def xyz2ply(input_xyz_path, output_ply_path):
    points = np.loadtxt(input_xyz_path)
    points, _, _ = normalize_points(points)
    num_points = points.shape[0]
    with_rgb = (points.shape[1] == 6)

    if with_rgb:
        points = [(points[i,0], points[i,1], points[i,2], points[i,3], points[i,4], points[i,5]) for i in range(num_points)]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(num_points)]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(output_ply_path)


def xyz2ply_with_rgb(input_xyz_path, output_ply_path, red, green, blue):
    points = np.loadtxt(input_xyz_path)
    points, _, _ = normalize_points(points)
    num_points = points.shape[0]
    
    points = [(points[i,0], points[i,1], points[i,2], red, green, blue) for i in range(num_points)]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(output_ply_path)


def get_file_ext(file_path):
    return file_path[file_path.rfind('.')+1:]

def get_file_name(file_path):
    basename = os.path.basename(file_path)
    return basename[:basename.rfind('.')]

def load_file(file_path):
    ext = get_file_ext(file_path)
    if ext == 'xyz':
        tensor = np.loadtxt(file_path)
    elif ext == 'npy':
        tensor = np.load(file_path)
    else:
        raise TypeError('file extension must be xyz or npy.')
    
    return tensor

def save_file(output_dir, basename, tensor):
    ext = get_file_ext(basename)
    save_file_path = os.path.join(output_dir, basename)
    if ext == 'xyz':
        np.savetxt(save_file_path, tensor, fmt='%.6f')
    elif ext == 'npy':
        np.save(save_file_path, tensor)
    else:
        raise TypeError('file extension must be xyz or npy.')

def clear_dir(dir_path):
    if not os.path.isdir(dir_path):
        return

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def clear_dir_with_time(root_dir_path, limited_time=1800):
    if not os.path.isdir(root_dir_path):
        return
    
    cur_ts = time()
    for root, dirs, files in os.walk(root_dir_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if cur_ts - os.path.getmtime(file_path) > limited_time:
                os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                if cur_ts - os.path.getmtime(dir_path) > limited_time:
                    os.rmdir(dir_path)

def delete_files_and_dirs(path):
    if os.path.isfile(path):
        os.remove(path)
        return
    clear_dir(path)
    os.rmdir(path)


def zip_dir(zip_dirname, output_zip_path):
    zip = zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(zip_dirname):
        fpath = path.replace(zip_dirname, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def count_num_files_for_dir(path):
    num_files = 0
    if not os.path.isdir(path):
        return

    for root, dirs, files in os.walk(path, topdown=False):
        num_files += len(files)
    return num_files


# render
def create_sphere_at_xyz(xyz, colors=None, radius=0.010, resolution=3):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.compute_vertex_normals()
    if colors is None:
        sphere.paint_uniform_color([0x69/0xff, 0x69/0xff, 0x69/0xff])
    else:
        sphere.paint_uniform_color(colors)
    sphere = sphere.translate(xyz)
    return sphere

def generate_points_obj(input_file_path, output_file_path, colors=None, prog_record=False):
    ext = get_file_ext(input_file_path)
    if ext not in ['xyz', 'npy']:
        raise TypeError()
    
    if os.path.exists(output_file_path):
        raise FileExistsException()

    points = load_file(input_file_path)
    if len(points.shape) != 2 or points.shape[1] not in [3, 6]:
        raise ShapeException()
    
    points, _, _ = normalize_points(points)
    N = points.shape[0]
    C = points.shape[0]
    mesh = create_sphere_at_xyz(points[0], colors)

    for i in range(N):
        mesh += create_sphere_at_xyz(points[i], colors)

        if prog_record:
            STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'static')
            PCBASE_DIRNAME = 'mypcbase'
            prog_file_path = os.path.join(STATIC_DIR, PCBASE_DIRNAME, 'temp', 'prog.txt')
            if 0 == i and os.path.exists(prog_file_path):
                os.remove(prog_file_path)
            if (i+1) % 1000 == 0 or (i+1) == N:
                with open(prog_file_path, 'w') as f:
                    f.write(str(int((i+1)/N*100)))

    o3d.io.write_triangle_mesh(output_file_path, mesh)


# point operation
def normalize_points(points):
    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(points ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)
    points = points / furthest_distance
    return points, centroid, furthest_distance


# tensorflow operation
def upsample_points(input_points_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = CFG.gpu
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    points = load_file(input_points_file)

    with tf.Session(config=run_config) as sess:
        model = Model(CFG, sess)
        pred_points = model.test(points)
    return pred_points
