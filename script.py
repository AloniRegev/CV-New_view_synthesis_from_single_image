import math
import os

import sintel_io
import numpy as np
import cv2 as cv


def getDepthAndIntrinsics(imagePath):
    camPath = imagePath[:len(imagePath) - 3] + "cam"
    depthPath = imagePath[:len(imagePath) - 3] + "dpt"

    intrinsics = sintel_io.cam_read(camPath)
    depth = sintel_io.depth_read(depthPath)
    return intrinsics[0], depth


def getFromIntrinsics(i):
    fx = i[0][0]
    fy = i[1][1]
    cx = i[0][2]
    cy = i[1][2]
    return fx, fy, cx, cy


def generatePoseFromKRT(K, R, T):

    RT = [R[i] + [T[i]] for i in range(len(T))]
    P = np.matmul(K, RT)

    # identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # RT = np.matmul(R, [identity[i] + [-T[i]] for i in range(len(T))])
    print("R: ", R)
    print("\nT: ", T)
    print("\nRT: ", RT)
    print("\nIntrinsics: ", K)
    print("\nPose: ", P)
    return P


def img_2d_3d(img, i, d):    #

    r, c = img.shape[:2]
    fx, fy, cx, cy = getFromIntrinsics(i)
    # points = [[x, j, 1] for x in range(r) for j in range(c)]
    # new_points = project_2d_3d(points, d, i)

    new_points = [project_2d_3d((i,j),fx, fy, cx, cy,d) for i in range(r) for j in range(c)]
    return new_points


# def project_2d_3d(points, depth, intrinsics):
#     invI = np.linalg.inv(intrinsics)
#     pts = invI[:3,:3] @ np.transpose(points) * depth.flatten()
#     pts = np.transpose(pts)
#     return pts

def project_2d_3d(p, fx, fy, cx, cy, d):
    x = p[0]
    y = p[1]
    z = d[x][y]
    new_x = (y - cx) * d[x][y] / fx
    new_y = (x - cy) * d[x][y] / fy
    return new_x, new_y, z

def project_3d_2d(point, P):
    pointz = np.transpose([point[0], point[1], point[2], 1])
    new_point = np.matmul(P, pointz)
    toRet = [int(new_point[0] / new_point[2]), int(new_point[1] / new_point[2])]
    return toRet

def rotation_matrix_x(theta):
    arr = [[1, 0, 0],
           [0, np.cos(theta), -np.sin(theta)],
           [0, np.sin(theta), np.cos(theta)]]
    return arr

def rotation_matrix_y(theta):
    arr = [[np.cos(theta), 0, np.sin(theta)],
           [0, 1, 0],
           [-np.sin(theta), 0, np.cos(theta)]]
    return arr
def rotation_matrix_z(theta):
    arr = [[np.cos(theta), -np.sin(theta), 0],
           [np.sin(theta), np.cos(theta), 0],
           [0, 0, 1]]
    return arr
def rotation_matrix_none(theta):
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
def translation_matrix_x(theta):
    return [theta, 0, 0]
def translation_matrix_y(theta):
    return [0, theta, 0]
def translation_matrix_z(theta):
    return [0, 0, theta]
def translation_none(theta):
    return [0, 0, 0]


def run_script():
    images = ["./inputs/ambush_6.png", "./inputs/alley_2.png","./inputs/market_2.png"]
    transformations = [
        (rotation_matrix_none, translation_matrix_x),
        (rotation_matrix_none, translation_matrix_y), (rotation_matrix_none, translation_matrix_z),
        (rotation_matrix_x, translation_none), (rotation_matrix_y, translation_none),
        (rotation_matrix_z, translation_none),
    ]
    degrees = [i for i in range(0,-16,-3)] + [i for i in range(0,16,3)]

    for imagePath in images:
        img = cv.imread(imagePath)
        r, c = img.shape[:2]

        counter = 0
        output_path = imagePath[:len(imagePath)-4] + "/"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # Each image should have .cam and .dpt files under the same name in the same folder.
        intrinsics, disparity = getDepthAndIntrinsics(imagePath)
        # Get 3d point-cloud
        points = img_2d_3d(img, intrinsics, disparity)

        # Get intrinsics of camera
        fx, fy, cx, cy = getFromIntrinsics(intrinsics)

        # Possible degrees for rotation.
        for transformation in transformations:
            R, T = transformation[0], transformation[1]
            images = []
            # Generate the new pose
            for d in degrees:
                counter += 1
                #output_file = output_path + str(counter) + "-rotation.png"
                output_file = output_path + str(counter) + ".jpg"
                if os.path.exists(output_file):
                    continue
                theta = np.radians(d)
                rotation = R(theta)
                translation = T(theta)
                P = generatePoseFromKRT(intrinsics, rotation, translation)
                new_img = np.zeros(img.shape, dtype=np.uint8)
                new_img_points = [project_3d_2d(point, P) for point in points]
                disparityImg = np.zeros(img.shape[:2], dtype=np.uint8)
                for i in range(len(new_img_points)):
                    # np = new image pixels
                    # p = RGB origin pixel
                    newp = new_img_points[i]
                    p = (int( (points[i][0] * fx / points[i][2]) + cx), int((points[i][1] * fy / points[i][2]) + cy))
                    if 0 <= newp[0] < c and 0 <= newp[1] < r:
                        if new_img[newp[1]][newp[0]].any():
                            if disparity[p[1]][p[0]] < disparityImg[newp[1]][newp[0]]:
                                disparityImg[newp[1]][newp[0]] = disparity[p[1]][p[0]]
                                new_img[newp[1]][newp[0]] = img[p[1]][p[0]]
                        else:
                            disparityImg[newp[1]][newp[0]] = disparity[p[1]][p[0]]
                            new_img[newp[1]][newp[0]] = img[p[1]][p[0]]

                # write new_img to proper folder :)
                print(output_file)
                cv.imwrite(output_file, new_img)
                images.append(new_img)
                if d == 15 or d == -15:
                    images = images[::-1]
                    for i in images:
                        counter += 1
                        output_file = output_path + str(counter) + ".jpg"
                        print(output_file)
                        cv.imwrite(output_file, i)
                    images = []



if __name__ == "__main__":

    run_script()