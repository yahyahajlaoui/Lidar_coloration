import cv2
import numpy as np
from pyntcloud import PyntCloud
import os
import fnmatch
from tqdm import tqdm
from pprint import pprint
import time
import open3d as o3d
def display_image(window, img):
    cv2.namedWindow(window, 0)
    cv2.imshow(window, img)

def find_matching_files(d, p):
    file_list = []
    for root, _, files in os.walk(d):
        for basename in files:
            if fnmatch.fnmatch(basename, p):
                filename = os.path.join(root, basename)
                file_list.append(filename)
    return file_list

def load_camera_to_camera_calibration(file, debug=False):
    with open(file) as f_calib:
        lines = f_calib.readlines()

    r_rect = []
    p_rect = []

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-4] == "R_rect":
            r_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r_r = np.reshape(r_r, (3, 3))
            r_rect.append(r_r)
        elif title[:-4] == "P_rect":
            p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            p_r = np.reshape(p_r, (3, 4))
            p_rect.append(p_r)

    if debug:
        print("R_rect:")
        pprint(r_rect)

        print()
        print("P_rect:")
        pprint(p_rect)

    return r_rect, p_rect

def load_lidar_to_camera_calibration(file, debug=False):
    with open(file) as f_calib:
        lines = f_calib.readlines()

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-1] == "R":
            r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r = np.reshape(r, (3, 3))
        if title[:-1] == "T":
            t = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            t = np.reshape(t, (3, 1))

    tr = np.hstack([r, t])
    tr = np.vstack([tr, np.array([0, 0, 0, 1])])

    if debug:
        print()
        print("Tr:")
        print(tr)

    return tr

def load_calibration_data(file, debug=False):

    with open(file) as f_calib:
        lines = f_calib.readlines()
    
        p_rect = []    
    for line in lines:
        title = line.strip().split(' ')[0]
        if len(title):
            if title[0] == "R":
                R_rect = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                R_rect = np.reshape(R_rect, (3,3))
            elif title[0] == "P":
                p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                p_r = np.reshape(p_r, (3,4))
                p_rect.append(p_r)
            elif title[:-1] == "Tr_velo_to_cam":
                Tr = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                Tr = np.reshape(Tr, (3,4))
                Tr = np.vstack([Tr,np.array([0,0,0,1])])
    
    return R_rect, p_rect, Tr

def load_image_data(file, debug=False):

    img = cv2.imread(file)
    
    if debug: display_image("Image", img)

    return img

def load_lidar_data(file, debug=False):

    points = np.fromfile(file, dtype=np.float32)
    points = np.reshape(points, (-1,4))
    points = points[:, :3]
    points.tofile("./temp_pc.bin")

    # Remove all points behind image plane (approximation)
    cloud = PyntCloud.from_file("./temp_pc.bin")
    cloud.points = cloud.points[cloud.points["x"]>=0]
    points = np.array(cloud.points)

    if debug:
        print (points.shape)

    return points

def calculate_projection_matrix_raw(file_c2c, file_l2c, cam_id, debug=False):

    # Load Calibration Parameters
    R_rect, P_rect = load_camera_to_camera_calibration(file_c2c, debug)
    tr = load_lidar_to_camera_calibration(file_l2c, debug)

    # Calculation
    R_cam2rect = np.hstack([R_rect[0], np.array([[0],[0],[0]])])
    R_cam2rect = np.vstack([R_cam2rect, np.array([0,0,0,1])])
    
    P_lidar2img = np.matmul(P_rect[cam_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print ()
        print ("P_lidar2img:")
        print (P_lidar2img)

    return P_lidar2img


def calculate_projection_matrix(calib_file, cam_id, debug=False):
    # Load Calibration Parameters
    R_rect, P_rect, tr = load_calibration_data(calib_file, debug)

    # Perform Calculation
    R_cam2rect = np.hstack([R_rect, np.array([[0],[0],[0]])])
    R_cam2rect = np.vstack([R_cam2rect, np.array([0,0,0,1])])

    P_lidar2img = np.matmul(P_rect[cam_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print()
        print("P_lidar2img:")
        print(P_lidar2img)

    return P_lidar2img

def project_lidar_to_image(image, pt_cloud, proj_matrix, debug=False):

    # Dimension of data & projection matrix
    dim_norm = proj_matrix.shape[0]
    dim_proj = proj_matrix.shape[1]

    # Do transformation in homogenuous coordinates
    pc_temp = pt_cloud.copy()
    if pc_temp.shape[1]<dim_proj:
        pc_temp = np.hstack([pc_temp, np.ones((pc_temp.shape[0],1))])
    points = np.matmul(proj_matrix, pc_temp.T)
    points = points.T

    temp = np.reshape(points[:,dim_norm-1], (-1,1))
    points = points[:,:dim_norm]/(np.matmul(temp, np.ones([1,dim_norm])))

    depth_max = np.max(pt_cloud[:,0])

    # Plot
    if debug:
        depth_max = np.max(pt_cloud[:,0])
        for idx,i in enumerate(points):
            color = int((pt_cloud[idx,0]/depth_max)*255)
            cv2.rectangle(image, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)
            print('---')
        cv2.imwrite("xxxxx.png", image)
    return points

def generate_color_point_cloud(img, pt_cloud, projected_image, debug=False):

    x = np.reshape(projected_image[:,0], (-1,1))
    y = np.reshape(projected_image[:,1], (-1,1))
    xy = np.hstack([x,y])

    point_cloud_color = []
    for idx, i in enumerate(xy):
        if (i[0]>1 and i[0]<img.shape[1]) and (i[1]>1 and i[1]<img.shape[0]): 
            bgr = img[int(i[1]), int(i[0])]
            p_color = [pt_cloud[idx][0], pt_cloud[idx][1], pt_cloud[idx][2], bgr[2], bgr[1], bgr[0]]
            point_cloud_color.append(p_color)
    point_cloud_color = np.array(point_cloud_color)

    return point_cloud_color


def save_pointcloud_file(file, pt_cloud_color):

    f = open(file, "w")

    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z rgb\n")
    f.write("SIZE 4 4 4 4\n")
    f.write("TYPE F F F F\n")
    f.write("COUNT 1 1 1 1\n")
    f.write("WIDTH {}\n".format(pt_cloud_color.shape[0]))
    f.write("HEIGHT 1\n")
    f.write("POINTS {}\n".format(pt_cloud_color.shape[0]))
    f.write("DATA ascii\n")

    for i in pt_cloud_color:
        f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(i[0],i[1],i[2],i[3],i[4],i[5]))
    
    f.close()

CALIB_TYPE = 0      # 0:All parameters in one file. e.g. KITTI    1: Separate into two files. e.g. KITTI raw
# if CALIB_TYPE == 0
CALIB = "./calib/000000.txt"

# Source File
IMG_PATH = "./img/"
LIDAR_PATH = "./lidar/"

# Save File
SIMG_PATH = "./result/img/"
SPC_PATH = "./result/pcd/"



CAMERA_ID = 2

def coloring(IMG_PATH,LIDAR_PATH,SIMG_PATH,CALIB,SPC_PATH,CAMERA_ID,CALIB_TYPE):
    time_cost = []

    # Calculate projection_matrix
    projection_matrix = calculate_projection_matrix(CALIB, CAMERA_ID)

    # Batch Process
    for image_path in tqdm(find_matching_files(IMG_PATH, '*.png')):
        _, image_name = os.path.split(image_path)
        point_cloud_path = LIDAR_PATH + image_name[:-4] + '.bin'
        # print ("Working on", image_name[:-4])        
        

        # Load image & point cloud
        image = load_image_data(image_path)
        point_cloud = load_lidar_data(point_cloud_path)
        start_time = time.time()
        # Project & Generate Image & Save
        points = project_lidar_to_image(image, point_cloud, projection_matrix)

        pcimg = image.copy()
        depth_max = np.max(point_cloud[:, 0])
        for idx, i in enumerate(points):
            color = int((point_cloud[idx, 0] / depth_max) * 255)
            cv2.rectangle(pcimg, (int(i[0]-1),int(i[1]-1)), (int(i[0]+1),int(i[1]+1)), (0, 0, color), -1)

        
        cv2.imwrite(SIMG_PATH+image_name, pcimg)

        # Generate Point Cloud with Color & Save
        point_cloud_color = generate_color_point_cloud(image, point_cloud, points)

        save_pointcloud_file(SPC_PATH + image_name[:-4] + ".pcd", point_cloud_color)
        
        # Time Cost
        end_time = time.time()
        time_cost.append(end_time - start_time)
    print ("Mean_time_cost:", np.mean(time_cost))
    return point_cloud_color

    print("Mean_time_cost:", np.mean(time_cost))
    cv2.destroyAllWindows()

def visualize_point_cloud(pc_xyzrgb):

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    if pc_xyzrgb.shape[1] == 3:
        o3d.visualization.draw_geometries([pc])
        return 0
    if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
        pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
    else:
        pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
    o3d.visualization.draw_geometries([pc])
    return 0

def read_pcd_file(path):

    xyzrgb = []
    with open(path, 'r') as f:
        content = f.readlines()
        for i in content[10:]:
            i_content = i.split(" ")
            x, y, z = float(i_content[0]), float(i_content[1]), float(i_content[2])
            r, g, b = float(i_content[3]), float(i_content[4]), float(i_content[5][:-1])

            xyzrgb.append([x, y, z, r, g, b])

    return np.array(xyzrgb)


if __name__ == '__main__':
    CALIB_TYPE = 0      # 0:All parameters in one file. e.g. KITTI    1: Separate into two files. e.g. KITTI raw
    # if CALIB_TYPE == 0
    CALIB = "./calib/000000.txt"
    # Source File
    IMG_PATH = "./img/"
    LIDAR_PATH = "./lidar/"
    # Save File
    SIMG_PATH = "./result/img/"
    SPC_PATH = "./result/pcd/"
    CAMERA_ID = 2

    pcd = coloring(IMG_PATH,LIDAR_PATH,SIMG_PATH,CALIB,SPC_PATH,CAMERA_ID,CALIB_TYPE)
    visualize_point_cloud(pcd)
